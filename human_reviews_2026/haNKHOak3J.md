# When sufficiency is insufficient: Probabilistic neural representations as an information bottleneck

- Avg Score: 4.50
- Decision: Reject
- Scores: 2, 6, 8, 2

## Abstract
The neural basis of probabilistic computations remains elusive, even amidst growing evidence that humans and other animals track their uncertainty. Recent work has proposed that probabilistic representations arise naturally in task-optimized neural networks trained without explicitly probabilistic inductive biases. However, prior work has lacked clear criteria for distinguishing probabilistic representations---those that perform transformations characteristic of probabilistic computation---from heuristic neural codes that merely reformat inputs. We propose a novel information bottleneck framework, the \textit{functional information bottleneck (fIB)}, that crucially evaluates a neural representation based not only on its statistical sufficiency but also on its \textit{minimality}, allowing us to disambiguate heuristic from probabilistic coding. To demonstrate the power of this framework, we study a variety of task-optimized neural networks that had been suggested to develop probabilistic representations in earlier work: networks trained to perform static inference tasks (such as cue combination and coordinate transformation) or dynamic state estimation tasks (Kalman filtering). In contrast to earlier claims, our minimality requirement reveals that probabilistic representations fail to emerge in these networks: they do not develop minimal codes of Bayesian posteriors in their hidden layer activities, and instead rely on heuristic input recoding. Therefore, it remains an open question under which conditions truly probabilistic representations emerge in neural networks. More generally, our work provides a stringent framework for identifying probabilistic neural codes. Thus, it lays the foundation for systematically examining whether, how, and which posteriors are represented in neural circuits during complex decision-making tasks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a test for emergent probabilistic representations in task-optimized neural networks, responding directly to Orhan and Ma 2017. They suggest that a probabilistic representation must be sufficient and minimal (invariant to nuisance variables). They then attempt to test whether simple networks optimized for simple tasks are indeed sufficient and minimal. They appeal to a "functional" information bottleneck concept, but instead of evaluating information, they use decoding approaches and compare decoding performance to two comparisons, a “copycat” network and a probabilistic population code (PPC), which should be, respectively, sufficient and minimal. They find that their trained networks are not minimal, and conclude that task-optimized networks do not represent probabilities.

### Strengths
The paper addresses an important question about what constitutes a probabilistic representation. They appeal to clear principles, and run appropriately simple tests. The probes are a reasonable way to approximate information content. Their results are interesting.

### Weaknesses
Giant problem:

They vastly overinterpret their results, and draw a much more dramatic conclusion than is warranted.
Their core premise is wrong: It is not a reasonable requirement that a probabilistic representation in the brain must be minimal. I'll give a few examples of why.

Let’s say a population encodes both p(x) AND log p(x), perhaps because the brain wants to marginalize sometimes and integrate cues sometimes. Or perhaps I encode some ancillary statistics in addition to the target distribution. These examples are both reasonable non-minimal representations of probability.

Their notion of minimality also breaks as soon as we consider pretty mundane population codes.

First, consider the whole brain. Then you could pretty clearly say that it is not a minimal representation of probability.

Second, consider only a subset of neurons, and evaluate this subset for their notion of sufficient and minimal. It seems their argument could conceivably work in such a localized code, treated in isolation.

Third, consider a multiplexed code. One subspace is encoding a probability distribution beautifully, but another orthogonal subspace in the same neurons is encoding something else. Then the authors' metric would say there is NOT a probabilistic representation because it’s not minimal, even though it's just a rotation of the second example of neuronal activity that were just deemed a probabilistic representation. 

Because their premise is so problematic, it undermines the entire paper. If they dialed back their claim to be what they show — task-optimized networks do not compress optimally — then it would be a valid conclusion but not a strong one.




Medium problems:

In the Kalman filter, the copycat baseline was supposed to keep all inputs, but now they only use a subset in a sliding window. So it’s not copycat now. I don't understand how this is a useful comparison anymore. I can understand a sliding window truncation of the task-optimized filter, which should learn to compress. And the truncated Kalman filter is truncated at a long enough window so it has only a tiny error. But the copycat cannot be truncated this way without losing even approximate sufficiency.


Invariance to nuisances is directly related to properties of decoder. For example, if you use a square decoder, then it doesn’t matter that the code is invariant to sign.



Minor comments:

Figure 3: Where is r? Is this h? x?

Is this nu the noise VARIANCE? Above, Q is called process noise but it’s actually the process noise variance, I believe.

Figure 4 caption: I think they mean the prior is over the sum of uniformly distributed variables, p(z1+z2) which would be triangular, not that the distribution is a sum of distributions, which would not be triangular.

"Apparent parity with the PPC is misleading because of the log scale" — not sure that this is a significant difference at all. And if there IS parity with PPC, then it undermines their claims that their network is not minimal.

### Questions
Can you justify why minimality is a criterion for a probabilistic representation?

More concretely: Say I have an input i, a probability distribution q based on that input, and a localized, sufficient, minimal neural representation r_q of that probability distribution q. Let's then concatenate the input and r_q, and then rotate the combination (i, r_q) by some rotation matrix R to give a vector z = R.(i, r_q), then doesn't the population z have perfect information about the input, plus perfect information (that is perfectly formatted for a linear decoder) about the resultant probability? This is not minimal, correct? But would you agree that z is a representation of the probability?

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper addresses question in computational neuroscience: when can neural representations be considered genuinely probabilistic rather than heuristic recodings of input. The authors propose the functional Information Bottleneck (fIB) as a principled criterion for distinguishing minimal, sufficient probabilistic representations from non-minimal ones. They apply this framework across several canonical perceptual and inference tasks and challenge earlier conclusions claiming that probabilistic representations emerge naturally in task-trained neural networks.

### Strengths
The paper makes a clear and meaningful conceptual contribution by introducing the functional Information Bottleneck (fIB) as a more appropriate criterion for identifying when neural representations can be interpreted as genuinely probabilistic. The motivation is strong and well-articulated, and the empirical setups are carefully chosen to revisit influential claims in computational neuroscience. The comparisons against copycat and explicit probabilistic population code benchmarks provide a convincing frame of reference, and the takeaway—that probabilistic representations do not reliably emerge under standard task optimization—is important and likely to shift ongoing discussions in the field.

### Weaknesses
However, the empirical evidence is drawn from relatively simple tasks and network settings, which may limit how broadly the conclusions can be generalized to deeper or more complex architectures common in modern machine learning. Additionally, relying on minimality as a decisive criterion may be seen as a strong assumption—some representations might remain useful or functionally probabilistic without being globally minimal. Finally, the approach depends on probe-based evaluations, which can introduce sensitivity to decoder capacity and training dynamics, raising questions about how stable the conclusions are under different probing choices.

### Questions
I have a few questions regarding the paper: 1) How robust are the FIB conclusions to changes in probe capacity or architecture? Since the evaluation relies on decoder probes, it would be helpful to know whether the same outcomes hold when using weaker/stronger probes or alternative decoding setups. 2) Do the findings generalize to more complex, higher-dimensional models or tasks?
The current experiments focus on relatively small networks and controlled inference tasks. Testing fIB on deeper networks or tasks with richer structure could strengthen claims about the generality of the conclusions.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper argues that neural networks trained on standard inference tasks don’t actually form true probabilistic representations of uncertainty. The authors introduce a framework called the functional Information Bottleneck (fIB), which tests whether hidden-layer representations are both sufficient (contain all task-relevant information) and minimal (contain nothing extra, especially related to inputs).
Using this method, they find that networks trained on cue combination, coordinate transformation, and Kalman filtering perform well but only recode inputs heuristically rather than compressing them into genuine Bayesian posteriors.

### Strengths
1/ Conceptual Clarity and Theoretical Innovation. The notion that probabilistic representation = sufficiency + minimality is elegant and clarifies an ambiguity in neuroscience and AI. Additionally, the fIB framework is a tractable, practical adaptation of the classic Information Bottleneck, avoiding the problems of estimating mutual information.

2/ Methodological Rigor. Thorough experimental design, with controls such as “copycat” and “probabilistic population code” benchmarks, strengthens the conclusions.

3/ Challenges the existing literature. The paper challenges a popular but weakly supported belief that probabilistic computation “emerges naturally” in neural nets. By showing the absence of minimality, it redefines what counts as genuine probabilistic representation.

### Weaknesses
1/ Limited empirical scope. The experiments focus on simple, low-dimensional tasks (cue combination, coordinate transformation, 1D Kalman filtering). It’s unclear whether the conclusions hold for more complex, high-dimensional networks or real sensory data.

2/ Dependence on probe networks. The fIB results depend on the behavior of trained probes, whose success depends on architecture and capacity. This makes it difficult to tell whether poor decoding reflects genuine information loss or simply probe limitations.

3/ No recovery (positive) case shown. The paper convincingly shows what doesn’t produce probabilistic representations but lacks an example where such representations do emerge. Without a successful counterexample, the framework’s diagnostic power remains one-sided.

### Questions
1/ Could minimal probabilistic representations emerge naturally if networks were trained with architectural bottlenecks, capacity limits, or noise regularization (e.g., dropout or variational objectives)?

2/ Recovery Cases. Related to the first point, what conditions or training regimes might allow a network to pass the fIB test — i.e., to develop representations that are both sufficient and minimal? Could explicitly probabilistic objectives or generative models serve as such recovery cases?

3/ Question about linear input probes. How sensitive are the fIB results to the choice and capacity of probe networks? Would defining minimality using linear rather than nonlinear input probes lead to different conclusions about what information is “functionally removed”?

4/ Broader Implications: If networks (and perhaps biological systems) can behave Bayesianly without encoding full probabilistic posteriors, what does that imply for how we interpret neural data and define “probabilistic computation” in the brain?

### Soundness
3

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
The paper proposes a quantitative criterion for qualifying internal representations of neural networks as "probabilistic" or "heuristic". For this purpose, internal representations in simple DNNs that perform cue integration, coordinate transformation and temporal filtering tasks are analyzed by "probe networks" that analyze whether the internal representation can be used to solve the problem, and/or to reconstruct the input. It concludes that the investigated DNNs do not form probabilistic internal representations without explicit additions to the loss function.

### Strengths
The paper tackles a long-standing problem, namely the encoding of information in biological and artificial NNs. It proposes a quantitative criterion to qualify an internal representation as "probabilistic" that could, in principle, be applied to any DNN. The experiments are well-described and the results support the claims that are made.

### Weaknesses
- The paper is easy to read but the message of the paper is not easy to understand. It is not made very clear what conceptual advantages a "probabilistic" internal representation could have, and whether it is a generally desirable thing to achieve in DNNs
- Likewise, it is not clear what the applicability of these results is. Like, would it improve classification results on MNIST if internal representations were probabilistic? Probably not, but what kinds of problem *could* be solved in this case is not made clear.
- The experiments are extremely simplistic, and so are the DNNs used. 
- The results, on very simple DNNs,  are essentially only negative, which is fine, but a wider investigation about how to *force* DNNs to develop probabilistic internal representations would complement this very well.

### Questions
- Please explain why the used networks and problems are so simple. Or rather: could all of this be applied to more complex problems, like, e.g., image classification? Or speech recognition? It seems that especially cue integration happens there as well, so this might be a far better motivation for this type of research
- Can the method described here be applied to any DNN? E.g., a LLM or ResNet-18?

### Soundness
2

### Presentation
1

### Contribution
2
