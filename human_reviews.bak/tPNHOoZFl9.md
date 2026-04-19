# Learning Dynamics of LLM Finetuning

- Decision: Accept (Oral)
- Scores: 6, 8, 8, 10

## Abstract
Learning dynamics, which describes how the learning of specific training examples influences the model's predictions on other examples, 
gives us a powerful tool for understanding the behavior of deep learning systems. We study the learning dynamics of large language models during different types of finetuning, by analyzing the step-wise decomposition of how influence accumulates among different potential responses. Our framework allows a uniform interpretation of many interesting observations about the training of popular algorithms for both instruction tuning and preference tuning. In particular, we propose a hypothetical explanation of why specific types of hallucination are strengthened after finetuning, e.g., the model might use phrases or facts in the response for question B to answer question A, or the model might keep repeating similar simple phrases when generating responses. We also extend our framework and highlight a unique ``squeezing effect'' to explain a previously observed phenomenon in off-policy direct preference optimization (DPO), where running DPO for too long makes even the desired outputs less likely. This framework also provides insights into where the benefits of on-policy DPO and other variants come from. The analysis not only provides a novel perspective of understanding LLM's finetuning but also inspires a simple, effective method to improve alignment performance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper analyzed the learning dynamics of LLM fine-tuning, in particular, the change of the loss function for other examples delta f(x_o) during training time parameter updates (from gradient descent with respect to input examples x_i), for both SFT and DPO. For SFT, the research used MNIST classification as an illustrative example to indicate that if x_o is similar to x_u but classified differently, the predicted probability of the x_u's class for x_o also got pulled up. For DPO, the research found that log likelihoods of both the positive examples y_u^+ and negative examples y_u^- all decreased, and the only example with log likelihood increase is the greedy decoding result y_u^*. This phenomenon could potentially explain the increased repetitiveness of DPO generated results towards the end of fine-tuning, and the increase of halluncinations during SFT. The researcher proposed a method to mitigate this effect by first SFT-ing on both positive examples and negative examples, and then perform DPO.

### Strengths
1. Analyzed the theoretical process behind DPO gradient updates. Explained deeply how the effect of gradient updates with respect to the training examples affect the model's performance on unrelated examples. Provided a theoretical framework for (potentially) understanding the origin of hallucinations.
2. Explained a common problem in DPO that was not illustrated well in prior work. For example, Rafael et.al. 2024 only vaguely claimed that "DPO decreases the likelihood of all data in favor of extrapolated responses", but this paper clearly indicated that the decreased likelihood all (or mostly) added to the greedy decoding sample (y^*).
3. The paper is well written and easy to follow.

### Weaknesses
Disclaimer: Only checked the main text and the Appendix A. Did not check Appendices B,C,D.

1. Need a "related work" section. The paper in its current form does not explain the relations between "learning dynamics" with related literature, for example, preventing negative transfers during transfer learning, neural tangent kernels, or loss landscape that enabled adversarial attacks. Offering a "related work" section whereas reducing the theoretical focus a little bit would help ICLR readers better position your work.

2. Need to indicate assumptions in your prop 1 and section 3.2 clearer. As far as I could see, there are at least 2 assumptions: 
- Higher order terms has little to no impact to learning dynamics
- K changes slowly during the initial training stages and thus has a limited impact to training dynamics

Please list out these assumptions (potentially what I did not listed out) so that users will follow and understand the limitations of your analysis.

3. Should design more experiments that 

- (a) During off-policy DPO, show that no other sentence's log probability significantly increased other than y^*. Although your theoretical analysis seems to have proved this, we still need to show this point in experiments due to your analysis assumptions. The sampling should be much more extensive to show the non-existence of such examples (and statistically significant)
- (b) For SFT, quantitatively show the causal effect of increased hallucinations due to the "pull up" effect of similar but negative examples (scaling up more than MNIST)
- (c) For both experiments, show that neuron tangent kernels (your K-term) did not change much, and thus the learning dynamics can be fully attributed to your G-term
- (d) Qualitatively demonstrate the repetitiveness of off-policy DPO towards the end of fine-tuning
- (e) Quantitatively measures how your proposed approach reduced hallucinations / answer repetitiveness via more rigorous benchmarks, other than just a measurement of the win rate

### Questions
1. In your page 5, "make any no assumptions" -> "make no assumptions"
2. In your appendix A, add more derivations to show that -(e_y_u^+ / pi)^T [A(x_u)]_l = pi(y_l | x_u) - e_y_u^+. All parts in Appendix A until this point is easy to follow.
3. In your page 16, lines 712-713, are there some nablas that are left behind?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The learning dynamics of a linear-softmax classifier on top of fixed features is analyzed. The method is applied to the settings of LLM finetuning, by considering particular loss/objective functions specific to this settings and analyzing their effect on learning dynamics. The results helps elucidating previously observed phenomenon in LLM finetuning, and constructing new methods for overcoming potential "bottlenecks" in the process.

### Strengths
The paper is overall well written, clearly explains the motivation and the approach. The use of simple "pedagogical" examples also helps in communicating the main message of the paper.

The main contribution seems to be in the application of the theory to the different loss functions used in LLM finetuning, making explicit use of the "decomposition" of the effective learning dynamics into 3 different terms, and identify which one directly depends on the loss function being used. This is demonstrated to be productive as it can provide (at least a preliminary) explanation for different observed behaviors.

Overall, this is an interesting paper and the results are likely to be of interest to the community.

### Weaknesses
There are two major limitations here that, while are (somewhat) acknowledged by the authors, can benefit from a more careful discussion and/or analysis.

1. The first is the obvious limitation that the entire analysis is being done under the assumption that the "feature map" is held fixed, and only the classification layer is changing during learning, but the extent to which this assumption holds in practice is not being evaluated at all. Without such an evaluation (for example, either by quantifying/tracking the change in $\mathcal{K}^t$, or by repeating some experiments when the weights are *actually* frozen) it is hard to say whether the theoretical explanation about the squeezing effect is in fact an important part of the observed phenomena (or whether it is largely / to some extent driven by more complicated dynamics that involves changes in the feature map as well). I think this should, at the very least, be acknowledged more explicitly in the text.
   
2. The second limitation is more nuanced, and has to do with the relying of some intuitive understanding of "similarity" (between prompts/examples), that may or may not be valid. Again no evaluation is offered -- despite the fact that the authors explicitly specify how this similarity should be understood (i.e., it is measured by the eNTK, $\mathcal{K}$ ). It is far from being clear a-priori that the learned features agree with some intuitive judgement of similarity. Moreover, what "dissimilar" means, in the extremely vast space of strings, is also non-trivial. For example (and as pointed out by the authors), it might be that all "well-formed"/grammatical sentences are somewhat similar, and the model is able to push probability mass to entirely non-grammatical/nonsensical strings. This leads to some arguments being made only on the basis of some intuition. For example, "the model’s confidence on y+u keeps increasing and the update energy... gradually decreases"  (Line 350) -- this seems to be inconsistent with the fact that we see no slow-down of the updates for $\mathbf{y}^+_{u}$ itself.

Finally I would also like to point out that peakiness (/"squeezing") effect in text generating models followed by RL tuning is a phenomena that has, in fact, been observed and to some reason explained before, including the observation that the on/off policy choice might play an important role in dealing with the "exposure bias". The paper will benefit from referring to some of these previous works, for example:
- Caccia et al., Language GANs Falling Short, (arxiv 2018 / ICLR 2020)
- Choshen et al., On the Weaknesses of Reinforcement Learning for Neural Machine Translation (arxiv 2019 / ICLR 2020)
- Kiegeland and Kreutzer, Revisiting the Weaknesses of Reinforcement Learning for Neural Machine Translation (naacl 2021)

### Questions
Line 31: "Many interesting phenomena" -- such as? (not saying there aren't, but this is an odd statement to have without providing at least some refs)

Line 61: I feel that this is conflating different things, i.e some notion of "interpretability" (some "compact" understanding of how model predictions/behavior are determined by the parameters), and notions of learning dynamics (how model parameters evolve during training). One can in principle discuss the one without the other, and it would be better to spell out more clearly / explicitly the current motivation.

Figure 1: The red arrows on panel a, top do not actually represent $\Delta\theta$ but rather $\Delta \pi$. It's also not clear why a continuous curve is plotted, given the fact that $\pi$ is discrete -- why not show all 10 digits as bars (as is later done in panels (c) and (d))? Perhaps most importantly, it's not clear whether or not this figure should be understood only qualitatively, or is this the result of an actual computation.
The same comments apply for all panels in Fig.1 (b). If this is an actual result then many details are missing, both in presentation (bars; scale of the y-axis) and perhaps more crucially, at what stage of learning this is done. The entire argument relies on some implicit assumption that "4" images are similar to "9" images, but this crucially depends on the current learned representation of the model (since this is a "feature-learning" rather than "lazy" regime). If this is just an illustration, then it is highly misleading presenting it along with panels (c) and (d) which seems to be actual numeric results.
I'm also unable to understand how to interpret panel (c). The caption says "accumulated change" but the the charts seems to indicate absolute probabilities. It's also unclear what is being tracked, over how many epochs, and what are the examples being presented. The main text (lines 134--138) is unhelpful and reads as another explanation of what's actually illustrated in panels (a)-(b) (i.e., the single-step influence)
Overall, the entire discussion of the motivating example in Section 2 is hard to follow. I think the overall intention of illustrating the idea in a simple example is a good one, but the example itself should be made much more clear.


Line 233: "Any no assumptions" -> "no assumptions" 


Line 253: Calling this "unexpected behavior" might be too strong, see previous literature
Line 314: "Fine grind" -> "Fine grained"

### Soundness
4

### Presentation
4

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
The paper extends the existing learning dynamics for deep learning systems to LLMs and  argues that doing so is a non-trivial problem due to high dimensionality and sequential nature of both inputs and outputs. The paper talks about a “squeezing effect” where running DPO for too many epochs causes outputs to deviate from ideal behavior.

### Strengths
1) The paper has strong theoretical and experimental backing for their analysis.
2) First paper to propose a framework extending learning dynamics to LLMs.

### Weaknesses
1) Did not find any specific weaknesses.

### Questions
How does the squeezing effect differ from overfitting with DPO?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
This work studies the learning dynamics of LLM finetuning by analyzing the way in which likelihoods assigned by the model to different completions of training prompts evolve throughout the finetuning process. By explicitly writing out the single step update rules in terms of the change in model parameters and change in the functional output of the model, and then leveraging the empirical NTK for the LM head/logit layer of the model, they show that one can analytically describe the sample wise influence dynamics of both the SFT and DPO objectives for LLMs. Corroborating prior observations of model distributions degenerating during (off-policy) DPO, their analysis describes a "squeezing effect" due to the performing of gradient _ascent_ during standard preference tuning, and they validate this and other qualitative predictions via the finetuning of 1B-3B parameter LLMs on carefully augmented preference tuning data.

### Strengths
1. They motivate and ground the study of training dynamics for LLMs well and highlight key difficulties in applying standard influence function style analyses to LLM finetuning and preference optimization, a highlight of the preliminary material.
2. The formalism and analysis in Section 3 is generally clear and well written. Figures 2 and the bullet point enumeration in 3.3 are concise presentations of the core claims of the analysis.
3. The experimental design of the empirical verification section is well done and cleanly visualized, particularly the use of completions $y^{+}$ gpts, gptf, test, hum.
4. Proposal for a mitigation to the "squeezing effect" (while not the main contribution of the paper) is well motivated by the analysis and simple to implement.

### Weaknesses
1. The clarity of Sec 3.1 when discussing the causal masking and teacher-forced production of the full next-token logits set could be improved, though this is admittedly tricky.  It is possible that for some readers, a diagram of the matrix structure here could be helpful since most papers don't tackle the more complex formulation of the influence problem and so readers may not be clear on it. (The reviewer imagines a teacher-forced causally masked model forward on input and label sequences X,Y as a forward on an augmented view of X,Y according to causal mask M, such that if the length of X+Y=N, the input has X' now has dim NxN (lower triangular consisting of all prefixes of the sequence X+Y) and we consider the output to be labels Y_i for each row of X' corresponding to the next token following the tokens in row X'_i.)

2. Additionally, while the analysis relies on the kernel values between the feature maps of $\mathcal{X_0}$ and $\mathcal{\tilde{X_u}}$ to implement the pressure transfer between various samples during learning, these intermediate values are not actually experimentally estimated. To concretely tie the analysis to the empirical observations, it would be useful to compute examples of the decomposition values for the terms in Eq 5 and Eq 7. If this intermediary relationship can be shown as clearly as the loss trends described in Figures 3 and 4, a stronger assertion of causality between the theoretical model described by Eq 5 and Eq 7 and the empirical learning dynamics can be made, strengthening the impact of the work.

### Questions
1. Can you discuss the analytic or empirical argument for "Peakier πθt suffer a more serious squeezing effect." in a bit more detail? I see arguments titles Claim 3A and 3B on pg 27, but they seem to be more intuitive that formal. It could be useful to separate the claims in the Sec 3.3 enumeration into those with direct proof/analytic evidence and those that are more informal or empirical -- this would not weaken the arguments, but rather making these subtle distinctions clear would increase the soundness of the presentation.

2. The color coding of the variables in section 2 and 3 is non-standard but fine if the authors like this (and no other reviewers mind it). However, it seems inconsistent to not continue this into Sec 4 and 5? Additionally, if there is a way to continue the scheme into the figures, especially Fig 2, that would make more full use of the fact that  the x and y's are colored in the related written sections.

### Soundness
4

### Presentation
4

### Contribution
4
