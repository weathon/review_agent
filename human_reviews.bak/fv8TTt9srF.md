# A Unified Theory of Quantum Neural Network Loss Landscapes

- Decision: Accept (Poster)
- Scores: 8, 3, 8, 8

## Abstract
Classical neural networks with random initialization famously behave as Gaussian processes in the limit of many neurons, which allows one to completely characterize their training and generalization behavior. No such general understanding exists for quantum neural networks (QNNs), which—outside of certain special cases—are known to not behave as Gaussian processes when randomly initialized. We here prove that QNNs and their first two derivatives instead generally form what we call "Wishart processes," where certain algebraic properties of the network determine the hyperparameters of the process. This Wishart process description allows us to, for the first time: give necessary and sufficient conditions for a QNN architecture to have a Gaussian process limit; calculate the full gradient distribution, generalizing previously known barren plateau results; and calculate the local minima distribution of algebraically constrained QNNs. Our unified framework suggests a certain simple operational definition for the "trainability" of a given QNN model using a newly introduced, experimentally accessible quantity we call the "degrees of freedom" of the network architecture.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This work establishes a unified theoretical framework for characterizing the trainability behavior of quantum neural networks. It integrates previous findings on barren plateaus and local minima phenomena by introducing a novel class of random processes.

### Strengths
- This paper establishes a unified framework for quantum nerual networks.

- It shows the necessary and sufficient conditions for QNNs to approach the Gaussian process limit, offering comprehensive insights into the trainability of these networks.

- While Jordan algebra is relatively unfamiliar in this domain, the authors have made efforts to clarify its application and relevance to the audience.

Overall, the results presented in this paper are impressive, and I believe they make a valuable contribution to the field of quantum machine learning.

### Weaknesses
- The paper should clearly show its contributions by more explicitly comparing what it offers beyond previous works. For example, it would be beneficial to clarify whether the results offer improvements in handling shallow circuits compared to previous studies in local minima.

### Questions
1. Is there an algorithm capable of automatically determining the trainability of any given QNNs?

2. Could you explain how to calculate $O^\alpha$? Are there any efficient algorithms for evaluating these calculations? It would be better if you can provide an example calculation or pseudocode for computing $O^\alpha$.

3. Previous work in [Anschuetz & Kiani,2022] analyzes local minima phenomena in local and deep, brickwall circuits. Does the current study address these issues? Furthermore, is it applicable to shallow circuits as well?

4. Does this work cover all previous results concerning BP and local minima?  If possible, it would be better to provide a more comprehensive comparison table or discussion section that explicitly shows how their framework subsumes or extends specific prior results on barren plateaus and local minima.

5. Could you clarify the rationale behind using Jordan algebra in this study?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
It is known that (classical) neural networks typically behave like Gaussian processes in the infinite-width limit under reasonable regularity assumptions. In this work, this result is extended to Quantum Neural Networks which behave like a Wishart process instead. This result is then used to study training, similar to Neural Tangent Kernels in the classical case, and the loss landscape thereof in particular with respect to barren plateus.

While the authors should be commended on their opus magnum and an exciting set of ideas / contributions from theoretical physics to machine learning, I share some concerns regarding technical soundness, presentation and topical fit to this publication venue.

### Strengths
1) Several new ideas are introduced from theoretical physics / quantum mechanics and connected with neural networks

2) The claimed result is a major generalization of major theorems in "classical" neural networks such as the infinite-width limit of neural networks (Gaussian processes) or Neural Tangent Kernel theory. While several results are derived from this for quantum NNs, there is potential that these ideas can also lead to new results for classical NNs, e.g. via the correspondence principle.

### Weaknesses
1) Technical: The paper lacks basic definitions and introduction of notation, or precision in doing so, that are crucial to follow the train of thought. This is detrimental as several parts of the notation are non-standard, or at least non-standard within the subsections and audiences of this publication venue. For example, in Theorem 1, a notation is introduced as "....denotes the projection into a (Jordan sub-algebra) A_α ....", first mention in the introduction around eq 2. Nowhere in the main body is it defined what this precisely means. QM people are left with a vague idea, but not more. Instead, the reader must dig deep into the appendix. It was until around eq 178 in the appendix until I got a better idea, by implication only. Further examples of lack of definition are around eq 5, 7 and 13, or mathematical objects \mathcal{G} and \mathcal{H} in Line 291. An example for unclear notation is also in eq 21 in Theorem 2. While non-standard notation is perfectly fine, it doesn't help much if things are kept too informal, even when things "are easy to see", and it should defined and introduced properly. This list is non-exhaustive.

2) Presentation: When agreeing to review, I was expecting a 10-page paper in line with the ICLR submission guidelines. While this work technically fulfills these requirements, one can learn and understand very little without working through the 45 page appendix. More honestly, this is a 59-page paper that has been last minute broken into a 10 page teaser with several big claims, many with unclear or vague rationale, followed by the actual paper.

Minor:
I know this publication venue typically expects numerical experiments, of which there is a lack of any in this paper. From a theoretical physics perspective, this is fine though.

### Questions
what is the potential of classical limits within this framework?

### Soundness
2

### Presentation
1

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The authors present a theory study of quantum neural networks showing, in contrast to classical neural network studies, that QNNs obey a so-called Wishart process. In doing so, they derive the existence of barren plateaus and show when a QNN behaves like a Gaussian process.

### Strengths
The paper is written very well and tackles a valid problem. The theoretical analysis appears to be well performed and the authors have done a lot to highlight the connection to prior results.

### Weaknesses
The main weakness I see is in the discussion around limits. It isn't always clear in what limit the authors are discussing the behaviour of the networks. It is also reasonable that a classical neural network's neural tangent kernel matrix has eigenvalues behaving a Wishart distribution and evolving during training. Further, a covariance matrix used to describe a GPR will likely also look very Wishart. While the math and results were presented in the paper, it was never completely clear to me how this all aligned and how I should think about the WIshart process.

### Questions
* Where does entanglement fall into these results? One criticism of quantum ML is that entanglement appears not to provide much a benefit in real-world problems. Is there an element of the theory presented here that can touch on this problem?

* What exactly should follow a Wishart distribution in the QNN framework? In a neural network, it is the weights, whereas the NTK will likely look Wishart and the Fisher matrix the same. Purely looking at the math, the NTK and Fisher matrix of a QNN should do the same.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The article defines a quantum neural network (QNN) as a chain of unitary operations applied to an initial quantum state. The loss function is the expectation value of an operator the resulting state with respect to an operator. This definition is common and general. The authors then assume random initialization and show that the loss converges to what they term a Wishart process in the limit of infinitely wide networks. This is in contrast to classical neural nets, where the limit is a Gaussian process (GP).   Having established the Wishart process framework, they use it to formulate several results within it:

(1) While the possibility of QNN convergence to GPs had already been studied in prior cited work, the authors formulate conditions within this framework for the convergence.

(2) Furthermore, they extend and generalize two results on trainability of neural networks from cited articles: 
- Conditions for barren plateaus, i.e. vanishing gradients.
- The density of local minima is concentrated near the global minimum iff the network is over-parameterized.

### Strengths
- The article is very well structured and seems technically of excellent quality. (However, I did not check or verify the proofs in the Appendix in detail) 
- The overall question on the structure of quantum neural networks and their potential advantages is relevant and the new findings advance the field.
The new Wishart process framework introduced in this article seems useful, as demonstrated by the results derived mentioned in the summary.

### Weaknesses
- The results are not groundbreaking new but generalizations of previous results and formulated in a more general language and setting.
- The results are a bit underwhelming, essentially "it seems unlikely that there exists any computational quantum advantage during the training of QNNs", which already seemed unlikely before. But this is expected, and this evidence is also valuable, so this is not a strong weakness.
- The main text is hard to understand without the long Appendix. However, this seems inevitable given the page limit and the subject.

### Questions
Can you give more explanation and background on N (equation 25) in the main text?

### Soundness
4

### Presentation
3

### Contribution
2
