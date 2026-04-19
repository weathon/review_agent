# On the Relation between Gradient Directions and Systematic Generalization

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 3

## Abstract
Systematic generalization is a critical property that most general deep learning algorithms lack. In this paper, we investigate the relation between gradient directions and systematic generalization. We propose a formulation to treat reducible training loss as a resource, and the training process consumes it to reduce test loss. We derive a bias that a training gradient is less efficient in using the resource at each step than an alternative gradient that leads to systematic generalization. The bias is avoided if and only if both gradients are zero or point in the same direction. We demonstrate the bias in standard deep learning models, including fully connected, convolutional, residual networks, LSTMs, and (Vision) Transformers. We also discuss a requirement for the generalization. We hope this study provides novel insights for improving systematic generalization.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper studies systematic generalization problems from the perspective of gradient directions. Based on the analysis of gradients on different distributions, the paper points out a bias that the training gradient is less efficient than another alternative gradient (the gradient on the ground-truth data distribution for all samples). Such a bias is experimentally verified on many classical deep neural networks, like CNN, LSTM, ViT, etc.

Although the idea of this paper is novel to me, the analysis and presentation of the paper are not good. It is hard for me to follow the paper, and I cannot see how the community can apply the proposed analysis to improve systematic generalization.

### Strengths
See the summary part.

### Weaknesses
Although studying the learning dynamics from the perspective of the gradient on different distributions (i.e., training, test, and overall) is novel, the paper lacks persuasive results and good presentations. It is hard for me to conclude the contribution of this paper. I have the following two main concerns. 

First, the systematic generalization (sys-gen) is not defined in this paper. What’s the difference between sys-gen and OOD problem? Are there any constraints on the differences between training and test distributions? What are the differences and similarities between training and test distributions in this setting?

Second, it is hard to see any potential of the provided analysis. The paper proposes several concepts based on gradient directions and then proposes a metric named UDDR. How is the UDDR gap related to systematic generalization, and how could we improve the systematic generalization performance based on the findings?

### Questions
1. Alternative direction is not a good term in my opinion. I guess this direction means the direction of the ground-truth data distribution.

2. In definition 3, “alternative gradient reduces the training loss”. I think this type of claim is a little weird. Because in practice, learning can only reduce the training loss, while the gradient for all data is unobservable. Better to say “training gradient reduces the alternative loss”. Or just call this "alternative gradient" an "oracle gradient", which means the optimal but inaccessible correct gradient.

3. Many of the concepts and definitions in this paper are based on vector inner produce, e.g., definition 3, 4, proposition 1, 2, etc. It is very hard to remember what they are discussing without a clear visualization. I think it would be helpful to visualize these concepts using a series of figures (similar to Figure 2) somewhere in the paper.

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper proposes a measure of out-of-distribution generalization bias based on a \textit{total} and \textit{training} gradient (cosine) similarity. After introducing definitions and lemmas, authors show (Theorem 1) that, letting alone degenerate case of zero gradients, the systematic generalization bias exists whenever  \textit{total} and \textit{training} gradients point in DIFFERENT direction, i.e., their cosine similarity is < 1. The claim is (partially) supported by experiments on the most popular deep learning architectures incl. transformers, LSTM and ResNet.

### Strengths
+ Well written Introduction section
+ Addressing a timely and needed research topic of systematic generalisation of deep learning models
+ Thorough build up of definitions and lemmas leading to the main Theorem 1 and related Propositions and Corollaries

### Weaknesses
While claim of Theorem 1 seems almost trivial from Definitions 1-4 (bias), the most of the paper covers formalism of this relation. But it rather misses explaining why would such a "local" step-wise definition of systematic generalization bias (see bellow for what is meant by "local") capture a "global",i.e., after all gradient descent steps are done, o.o.d. generalization gap. In my opinion the paper in its current form has made interesting initial steps but has not succeeded in showing the proposed measure of bias is a "good" one (in a sense described bellow).

- While experimental setup is described at length, the summary of experiments is only one paragraph long and lacking needed details. More over, Contributions, bullet 3 on page 2, it is claimed: “Experiments validate the result and demonstrate a bias in standard deep learning models ”. 

What is meant by “validate” exactly? The fact that there is generalization gap present in the most of DL models is generally known. If authors want to show that UDDR gap is good measure of it then they should "validate" not only UDDR gap =>"bad systematic generalization", but as well a contraposition the statement, i.e.,  "good o.o.d. generalization => low UDDR gap". The experiments do not show any such example to my knowledge.

- “We also discuss that systematic generalization requires a network decomposed to sub-networks, each with a seen test inputs. ” This is very interesting suggestion and possible research direction that would deserve a bit more comments perhaps. The section 4.2 only provides very brief and, to me, incomprehensible comments, however. Especially the Requirement on page 8, “ Systematic generalization requires that a model can be decomposed into sub- networks, each with seen test inputs. ”. For instance what is a “subnetwork with seen test inputs?”

- Proposed method of DDR (and D()) measures step-wise bias, i.e., a "local" bias at a given step of a gradient descent. How does this local bias relate to a final, "global", systematic generalization gap? Can there be two different pathways leading to the same model/result? Why not?

### Questions
In addition to questions raised in "Weaknesses" section. 

- Contributions: Claim No. 3 on page 2: “Experiments validate the result and demonstrate a bias in standard deep learning models ”. What is meant by “validate” exactly? The fact that there is generalization gap present in the most of DL models is generally known. If authors want to show that UDDR gap is good measure of it then they should "validate" not only UDDR gap =>"bad systematic generalization", but as well a contraposition the statement, i.e.,  "good o.o.d. generalization => low UDDR gap". The experiments do not show any such example to my knowledge.

- What is “sin” in Prop 1? Please add am explanatory note …

- Fig 4. UDDR of exactly what is depicted? It has three arguments … is if UDDR(test, train, all)?

-(Q): Fig 4, 5 Did both networks converge to same or different optima? Could you add these results?

More generally, how does proposed approach deal with stochastic gradient descent with noisy gradient and possibly several paths leading to the same solution? The appendix treats this very briefly and does not answer the question in my opinion.

### Soundness
2 fair

### Presentation
1 poor

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work explores how the alignment of the parameter update of a neural network followed by gradient descent with that of the gradient which aligns with the true (all data distribution) gradient. A number of cases are considered, which provable cover all possible outcomes, and it is shown that standard deep learning models are predominantly biased away from systematic generalisation. This argument is supported empirically for a range of practical models and tasks.

### Strengths
## Originality
The notion of treating the gradient update step as a resource which must be allocated is indeed interesting and an intuitive interpretation of events.

## Quality
The experimental design of Section 3 does support the main claim of the theory that neural networks are inherently biased away from generalisation. In addition the experiments cover a wide range of models that are of practical interest to the community. This makes the results interesting and useful. The claims which are made from these experimental results also appear accurate. On the theory side, definitions are clearly stated and used consistently throughout and propositions are formal but also explained clearly.

## Clarity
The work clearly describes the hypothesis. Figures are clear, interpretable and relate easily to the text.

## Significance
This work aims to address an interesting problem, and importantly does so for a broad range of practically useful models. Thus, the proposed finding of explaining why this broad range of models fail to systematically generalise could be of high impact. I do have some concerns on whether the work fully realises its intended purpose which I discuss in the Weaknesses section below. This limits the potentially high impact and significance of this work.

### Weaknesses
## Quality
My main issue with this work is that it does not seem to address systematic generalisation, but generalisation more broadly. A primary purpose of systematicity and compositional generalisation is to break the task into smaller pieces which are then learned (the modules specialise to the subtask) and composed later. By definition learning these smaller pieces is not the same as learning the entire data distribution. This seems at odds with this work's proposed theory that how far a network deviates from the all gradient direction is an indicator of systematicity. For example, learning a module which identifies the colour red, another which identifies cars and then learning to use both modules to identify red cars - as would be the case with neural module networks [1] - would follow extremely different gradient directions than if a similar network trained to directly identify all red cars. Put another way, to learn a disentangled representation [2] (a stronger condition than systematicity) you would need to follow a different gradient than if you learned the ground-truth mapping directly. Even from a linguistics perspective, seeing only a subset of data and this being used to learn something different from memorizing all of language is a foundational idea in systematicity [3]. Finally, more recent theory even explicitly makes the distinction between generalising because the network has seen enough data and generalising because the network decomposed the problem and learned a solution with an entirely different rank [4]. This last point is from a paper which is only a year old - and so open to debate - but this work would need to at least show how its definition of systematicity aligns with these prior notions. So for example the line "The alternative gradient is computed from all data, leading to systematic generalization" is just at odds with our notions of systematicity. This work, is of interest to generalisation broadly however, just not systematic generalisation as far as I can tell. Also the mention of "seen test inputs" is confusing and I don't know what this is meant to be referring to. But by definition test inputs are unseen.

## Clarity
On the point of clarity, there are some statements in this work which do not make sense or appear out of context. Examples are "Also, deep learning does not require many task-specific designs for specific tasks", "Some standard networks, such as ... work well in i.i.d. settings", "To keep the advantage, we  discuss whether standard deep learning models achieve systematic generalization", "The condition $\Delta=0$ is rare to hold because it requires an equation to hold" and "Both (A) and (B) contain equal signs, which are generally difficult to hold". Hopefully these examples will guide a general clean up of the writing.

A few more quicker points and concerns on clarity are the following. The last paragraph of page 2 where the notation is introduced is also not clear and introduces more notation than necessary. Why are $u$ and $h$ defined here and why have $u$ if $x$ already denotes input vectors? Similarly, $D(f,u)$ is defined and includes a case for if $u=0$ where on the top of the same page it is stated that $u \neq 0$. $\Delta$ is overall unhelpful as it obscures comparison with the other uses of  the $D(\cdot,\cdot)$ function. Definition 3 and 4 could be merged with Propositions 1, 2 and 3 since the propositions follow immediately from the definitions. Proposition 2 and 3 could also just be combined since the two cases are practically identical. DDR is also only used for two of five cases and so appears to be another function needlessly defined which just obscures the comparison of various cases. Also, why not use UDDR from the beginning? The captions for Figure 3 and Table 1 should be improved and clearly state why we should care about these datasets, how they are used and why they relate to systematicity.

[1] Andreas, Jacob, et al. "Neural module networks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016. \
[2] Locatello, Francesco, et al. "Challenging common assumptions in the unsupervised learning of disentangled representations." international conference on machine learning. PMLR, 2019. \
[3] Hadley, Robert F. "Systematicity in connectionist language learning." Mind & Language 9.3 (1994): 247-272. \
[4] Jarvis, Devon, et al. "On The Specialization of Neural Modules." The Eleventh International Conference on Learning Representations. 2022.

### Questions
I have asked a number of questions and raised some concerns in the Weaknesses section where they naturally came up. I do not currently have any further questions for this section but would appreciate if these early questions were addressed.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
