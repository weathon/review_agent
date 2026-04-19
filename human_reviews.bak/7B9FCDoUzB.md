# Regretful Decisions under Label Noise

- Decision: Accept (Poster)
- Scores: 8, 8, 6, 5

## Abstract
Machine learning models are routinely used to support decisions that affect individuals -- be it to screen a patient for a serious illness or to gauge their response to treatment. In these tasks, we are limited to learning models from datasets with noisy labels. In this paper, we study the instance-level impact of learning under label noise. We introduce a notion of regret for this regime, which measures the number of unforeseen mistakes due to noisy labels. We show that standard approaches to learning under label noise can return models that perform well at a population-level while subjecting individuals to a lottery of mistakes. We present a versatile approach to estimate the likelihood of mistakes at the individual-level from a noisy dataset by training models over plausible realizations of datasets without label noise. This is supported by a comprehensive empirical study of label noise in clinical prediction tasks. Our results reveal how failure to anticipate mistakes can compromise model reliability and adoption -- we demonstrate how we can address these challenges by anticipating and avoiding regretful decisions.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This work proposes an evaluation framework for noisy label learning methods in terms of "regret," as quantified by the discrepancy between model errors with respect to noisy labels vs. errors with respect to true labels. But regret is not distributed equally in the data: in their own words, "even if we can limit the number of mistakes, we cannot anticipate how they will be assigned over instances that are subject to label noise." The proposed approach takes a generative model of noise to train a set of models on plausible (under the distribution induced by the generative model) clean realizations, and estimates instance-level "regret" accordingly. Empirical results show that a common noisy-label learning baseline and naive approaches (ignore noise) exhibit non-zero regret consistently. A case study on a genomics dataset demonstrates the practical utility of the approach by leveraging an instance-level ambiguity measure derived from regret to abstain from low-confidence predictions.

### Strengths
* This is a very well-written paper. The prose is clear and the technical aspects of the problem motivation are well-defined and explained concisely. 
* The proposed approach is very simple and the theoretical results are intuitive, but backed by rigorous theoretical and empirical analyses. 
* Rather than assuming completely random models of noise, the proposed approach is adaptable to arbitrary generative models of noise.

### Weaknesses
* [W1 — knowing the true noise model]: The proposed approach requires knowledge of a full generative noise model of $(U, X, Y)$. It is not clear where this would come from in practice. This weakness is somewhat mitigated by the discussion at the end of Section 3 and empirical results showing robustness of the proposed approach to noise model misspecification, but building more formal machinery to characterize the sensitivity of the approach to noise model misspecification would strengthen the paper.  
* [W2] Proposition 4 provides the motivation for the proposed approach — using the generative model of noise, sample plausible realizations of the clean dataset. But the variance of the posterior could be extremely high — even with the $\varepsilon$-plausibility constraint (Def. 7), this could still yield high-variance regret/ambiguity averages. 
* [W3] I'm unsure about the usefulness of Prop. 5, which "implies that we can only expect hedging to learn a model that does not assign
unanticipated mistakes when $\mathbf{u}\_{mle} = \mathbf{u}\_{true}$. I read this as "models will overfit in finite samples to the observed noise draw rather than the true noise draw," which is intuitive. But if regret grows very, very slowly in $|\mathbf{u}\_{mle} - \mathbf{u}\_{true}|$ (any measure of distance between the two, to abuse some notation) — then it seems like this effect is not an issue. 
* [W4 — minor] The presentation of empirical results could be improved. Table 3 is very large, and it's hard for me to parse what I'm looking for. Similarly, Figures 3 and 4 could be designed a little more informatively — specifically, the caption should include a statement about why the proposed approach is "better" (e.g., our approach has X property, while the standard approach ... ).

### Questions
* Re: [W1] — I would love to hear any thoughts on the robustness of the proposed approach to noise model misspecification from a theoretical perspective.
* Re: [W2] — I would love to hear any commentary on how high-variance in the noise posterior could negatively affect the proposed approach. 
* Re: [W3] — Are small violations of the $\mathbf{u}\_{mle} = \mathbf{u}\_{true}$ condition (Prop. 5) truly "problematic?" Is there an example to demonstrate this? 

**Other questions/suggestions**
* Did the authors consider looking at metrics beyond expected regret/ambiguity (e.g., worst-case over $\varepsilon$-plausible models)?
* The noisy label evaluated in the experiments is >10 years old; while the value of the approach isn't based on which underlying noisy label learning method is under evaluation, it might be more salient to the noisy-label learning community to test a more recent suite of methods + different noise models. For example, [some](https://arxiv.org/abs/1809.03207) [methods](https://arxiv.org/abs/2406.18865) specify a full generative model and cast the clean label as a latent variable, while [other](https://arxiv.org/abs/2002.07394) [approaches](https://arxiv.org/abs/1910.01842) filter out examples flagged as noisy (according to some rule) in the learning process. Given the plethora of assumptions/noise models in the literature, I wouldn't be shocked if there is systematic variation in errors across methods. 

**Minor Suggestions**
* In Table 2, $\hat{\mu}(x)$ is defined as the median, but in Eq. (8), it is defined as the mean — I suggest making the definition consistent. 
* The proof of Prop. 4 in Appendix A is a little unclear: there are also some typographical inconsistencies (math mode vs. regular font), and I think $f(X)$ is mistakenly written as $X$ in one of the loss terms as L725-726. I was unable to replicate the final step, but this is likely since I had a hard time following the parentheses/whether each line was a continuation of the previous. Could this be clarified? I believe the result, since it seems to be essentially a result of the form E_{noise}[estimand] = estimand as is common in the noisy-label learning literature. 
* Is Prop. 9 (Appendix A only) not simply a restatement of Lemma 1 from [Learning with Noisy Labels, Natarajan et al., NeurIPS '13](https://proceedings.neurips.cc/paper_files/paper/2013/file/3871bd64012152bfb53fdf04b401193f-Paper.pdf)? If so, the proof can be omitted and replaced with the relevant citation. 
* Prop 10. and 11 appear to be standard applications of a weak law of large numbers + Hoeffding. If they're not referenced in the main text, consider removal.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors tackle the situation where observations come with label noise. They introduce a criterion (regret) which measures when the prediction errors disagree when the model is computed with  noisy observation \tile{y} and not noisy y. They develop a new method that estimate the posterior distribution of the possible noise and try to sample these observations. Hence if the distribution of the noise is well chosen, it becomes possible to construct set of plausible models and thus detect zones for which the uncertainty is above a certain level. The paper develop a new theory and provides sound mathematical proofs and simulations.

### Strengths
The point of view which is developed is interesting and is a valuable contribution.
The ideas are straightforward once the frame is set : defining the regret and then plausible sets minimizing the regret on epsilon-plausible datasets.
Experiments are convincing.
A whole section is devoted to the theoretical analysis of the results. Proposition 12 proposes the statistical guarantees of the methos.

### Weaknesses
I found the paper sometimes difficult to read and some sentences are difficult to understand 
l159 : "a practitioner may be able they expect ... " I can not understand what the authors mean.
l162 : the definition of the regret is fuzzy with some words that are not properly defined . "anticipated" mistake. What does it mean since you compare the error with labels with noise and labels without ?
l172 the paper should be self-contained if possible, so epxlain the comparison with \tilde{l}_{0,1}
l182 : what is " :-= " ?
l215: sometimes you use words that have a mathematical meaning : "most likely to flip" for instance

### Questions
can you define \tile{l}_{0,1}

### Soundness
3

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
This paper examines the problem of learning from noisy labels by addressing instance-level noise. The main contribution is the insight that a method performing well over the population can still lead to errors at the instance level. The paper introduces the concept of "regret" to characterize this phenomenon and proposes a method to mitigate the regret caused by randomness sampling multiple plausible noisy label draws. Theoretical analysis and experiments are presented to validate the proposed approach.

### Strengths
+ This paper addresses an important yet challenging task: learning from instance-level label noise.
+ Theoretical analysis and experiments are conducted to validate the proposed method.

### Weaknesses
- **Unclear Benchmark Algorithm**: One of my main concerns is the paper's clarity, particularly regarding the introduction of the benchmark algorithm critiqued in Section 2. It appears the paper intends to use a noise-tolerant method, such as that of Natarajan et al. [37], as a benchmark. However, by Proposition 5, the algorithm under discussion seems to actually refer to a different approach (let's call it Benchmark 2). In Benchmark 2, an implicit noise draw $\mathbf{u}^{\mathrm{mle}}$ is generated, $y_i$ is recovered using this noise draw, and then ERM is performed to train the model. To me, this algorithm (Benchmark 2) differs from the method in [37], especially in terms of instance-level performance. Therefore, it is less convincing that the criticisms for Benchmark 2 are applicable to the noise-tolerant method in [37]

- **Clarity on Notation**: I find the notation in Section 2 somewhat confusing, particularly in distinguishing which variables are random and which are deterministic. Based on the discussion in lines 130-135, it appears that $ y_i $ is deterministic, while $ U_i $ and $ \tilde{y}_i $ are random variables generated based on $ y_i $. However, I struggle to interpret the equation in line 173, as it seems $ y_i(U_i) $ is simply a deterministic value, making it challenging to see how it could be compared in inequality to a random variable.

- **Regarding Proposition 4 and its Proof**: It would be helpful if the authors provided a clearer explanation of which random variables the expectation is taken over. In the proof, it appears that the expectation is taken over $ X, \tilde{Y} $, and $ U $ while this is not mentioned in the main text. Additionally, I find it difficult to follow the reasoning in lines 734-744; the conclusion seems to rely on $ E_{X, \tilde{Y}, U}[\text{Regret}] $, yet the analysis is conducted for $ E_{X, Y, U}[\text{Regret}] $. Finally, the last lines indicate only that $ E_{X, \tilde{Y}, U}[\text{Regret}] > 0 $, but it is unclear how this leads to the conclusion stated in Proposition 4.

- **Strong Assumption**: The proposed method in Section 3 requires knowledge of $ P(U = 1 \vert X, Y) $. This is somewhat a strong assumption to me, as accurately estimating this value is generally challenging.

- **Insufficient Experiments**: It appears that the paper does not compare the proposed method with others in the literature. At a minimum, it would be beneficial to include a comparison with the noise-tolerant method [37] under the condition that $ P(U = 1 \vert X, Y) $ is known. Although [37] is designed for class-dependent noise, with knowledge of  $P(U = 1 \vert X, Y)$, extending it to handle instance-level noise should not be too challenging.

### Questions
- Could you elaborate further on the benchmark algorithm discussed in the paper and clarify its relationship with [37]?
- Could you provide an additional explanation regarding the proof of Proposition 4, particularly addressing the concerns mentioned in the weaknesses above?
- Could you include a performance comparison with other methods in the literature?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
The authors introduce the notion of regret when learning from a dataset that is subject to label noise. The authors point out that standard learning approaches typically target a notion of “average” loss or “average” risk over the population and cannot provide instance level guarantees. One way to identify that the model may make a mistake is to have access to clean labels, but this is often infeasible in practice. As a result, the authors propose to simulate “clean” datasets by assuming a noise model, simulating noise from that noise model, and then backing out a clean dataset from the noisy dataset and the sampled noise. Then the authors define a notion of ambiguity based on models trained on various plausible clean datasets.

### Strengths
This paper emphasizes that standard machine learning methods target some notion of “average loss” or “average risk” but that does not provide guarantees on performance for an individual instance, which is an important point. In particular, I enjoyed lines 186-188 “we cannot anticipate how [mistakes] will be assigned over instances that are subject to label noise. In this case, each instance where [there is a nonzero probability of a label flip] is subjected to a lottery of mistakes.”

The idea of constructing multiple plausible clean datasets from a noisy one is interesting, and seems very reminiscent of distributionally robust optimization (the idea constructing the set of plausible noise draws seems related to constructing a robustness set over distributions). It might be worthwhile to consider what connections there are between constructing the set of plausible noise draws and a robustness set.

### Weaknesses
- It would be helpful if the authors could provide additional explanation on how the notion of regret differs from standard classification accuracy, and why it is useful.

- A key limitation of the approach is that it requires the machine learning practitioner to specify a reasonable noise model. 

- The justification for restricting the sampled noise draws to a set of “plausible” noise draws is not clear to me. Why can’t we just account for the fact that each noise draw has a different likelihood? 

- The ambiguity quantity is not well-motivated. Why is it defined as the fraction of misclassifications across the cleaned datasets? How do the authors intend this ambiguity quantity to be used? Under what conditions is ambiguity equal to zero?

### Questions
Why do we write $y_{i}(U_{i})$ in the definition of $U_{i}$, from what I recall, $U_{i}$ is generated given $y_{i}$, so it is a bit confusing to think of $y_{i}$ as a function of $U_{i}$.

### Soundness
2

### Presentation
3

### Contribution
2
