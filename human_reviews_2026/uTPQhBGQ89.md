# Positive Distribution Shift as a Framework for Understanding Tractable Learning

- Decision: Reject
- Scores: 6, 2, 8

## Abstract
We study a setting where the goal is to learn a target function f(x) with respect to a target distribution D(x), but training is done on iid samples from a different training distribution D’(x), labeled by the true target f(x).  Such a distribution shift (here in the form of covariate shift) is usually viewed negatively, as hurting or making learning harder, and the traditional distribution shift literature is mostly concerned with limiting or avoiding this negative effect.  In contrast, we argue that such a distribution shift, i.e. training using D’ instead of D, can often be positive, and make learning easier, and that such a positive distribution shift (PDS) is central to contemporary machine learning, where much of the innovation in practice is in finding good training distributions D’, rather than changing the training algorithm.  We further argue that the benefit is often computational rather than statistical, and that PDS allows computationally hard problems to become tractable even using standard gradient-based training.  We formalize different variants of PDS, show how certain hard classes are easily learnable under PDS, and make connections with membership query learning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Typically, training on a different distribution $D'$ than the target distribution $D$ (a covariate shift) is seen as problematic, but the authors in this paper introduce Positive Distribution Shift (PDS): the idea that with a well-chosen $D'$, learning a target function on $D$ can actually become easier.
In other words, the paper argues that many computationally hard learning problems can be made tractable simply by picking a clever training distribution, while keeping the model and algorithm (e.g., standard SGD on a neural network) unchanged.

The authors provide a theoretical framework for PDS and demonstrate its power through examples. They formalize several variants of PDS learning, including cases where the training distribution may depend on the target function or not.
Using these definitions, they show that certain hypothesis classes which are computationally infeasible to learn under the same train-test distribution become efficiently learnable under an appropriate distribution shift.
These theoretical findings are complemented by empirical evidence: the authors train standard neural networks on strategically biased distributions and observe successful generalization to the original target distribution, confirming that PDS can markedly improve learning efficiency in practice.

Finally, the paper draws a connection between PDS and membership query learning.
It shows that designing a training distribution is analogous to a non-adaptive way of querying an oracle for labels, thereby unifying PDS with classical active learning concepts.

### Strengths
- **Originality:** The paper introduces a fresh viewpoint on distribution shift. Instead of treating distribution shift as a challenge to overcome, it is framed as a resource to exploit for easier learning. This positive framing of distribution shift is, to the best of my knowledge, original and insightful. It connects to the intuition behind practices like pre-training or curriculum learning, but provides a formal lens to understand them.
- **Theoretical contributions:** The paper is thorough in developing a theoretical foundation for PDS. It defines multiple variants of the PDS learning framework (function-dependent PDS, as well as deterministic and randomized distribution-shift PAC learning) with clear formalisms. Building on these definitions, the authors prove several non-trivial results. Notably, they show that classes previously deemed intractable under standard PAC learning become tractable with an appropriate choice of training distribution.
- **Clarity:** Overall, the paper is written clearly and is well-structured. The authors do a commendable job explaining the motivation and implications of PDS. In terms of writing, the technical content is dense but generally understandable, and the authors have provided intuition alongside formal statements (for example, explaining in words how a biased distribution reveals parity bits via correlations).
- **Significance:** The framework and results presented could have significant implications for learning theory and practice. By showing that “all functions are easy, with the right training distribution” (as posed in Section 3) in certain formal senses, the paper provides a possible explanation for the effectiveness of strategies like active learning, curriculum design, and synthetic data augmentation.

### Weaknesses
- **Reliance on strong or unrealistic assumptions in some settings:** Some of the theoretical formulations, particularly the function-dependent PDS (f-PDS) scenario, assume knowledge that would not be available in practice. In the f-PDS framework, the training distribution $D'$ is allowed to depend on the target function $f$ itself. This is a very powerful setting. Indeed, under f-PDS the authors note one can even learn all poly-size circuits with label noise (by encoding $f$ into $D’$), but it borders on a cheat, since if one knows enough about $f$ to craft $D'$, the learning problem becomes trivial. While the paper’s main focus soon shifts to more realistic variants (where $D'$ may depend on the target distribution $D$ but not on the unknown function $f$), the strongest positive results often stem from assumptions like f-PDS or the learner having partial knowledge (e.g. knowing the parity’s sparsity). This limits the direct applicability of those results.
- **Specialized training procedures for provable results:** In the cases where the paper tackles hard classes without assuming knowledge of $f$ (notably in the DS-PAC setting), the proposed learning methods sometimes rely on non-standard or tailored techniques. For example, the gradient-based algorithms used in the proofs are “stylized” in that they require $\ell_1$ regularization or carefully chosen hyperparameters that depend on the target function’s complexity (such as the sparsity $k$ of a parity). Theorem 4.5, for instance, shows PDS learnability of parity by gradient descent but only when the network is trained with a special initialization and regularization tuned to the parity’s $k$. This undermines, to a degree, the claim that standard SGD on a standard architecture suffices.
- **Limited scope of empirical evaluation:** The experiments provided, while valuable, are restricted to synthetic boolean function tasks (parities and juntas). These tasks are canonical in theoretical computer science, but they are far from realistic machine learning applications. It remains an open question how well the PDS framework would translate to, say, image classification, natural language, or other complex domains. The paper would be stronger if it at least discussed or hypothesized about this translation. For example, can we view certain data augmentation strategies or curriculum learning in deep learning as creating a “positive distribution shift”? The authors hint at this connection in motivation but do not demonstrate it. I encourage the authors to add at least a discussion section contemplating how one might *operationalize* PDS in practical settings (e.g. how to simulate the “well-chosen $D'$” when you have limited access to $D$). This would help readers appreciate the potential impact beyond curated boolean problems.

### Questions
- **Practical identification of $D’$:** In a real-world scenario, how would one go about finding a “good” training distribution $D'$ without knowing the target function in advance? The theoretical results often assume either an omniscient choice of $D'$ (in f-PDS, depending on $f$) or at least knowledge of the class’s structure (e.g. knowing that a parity has sparsity $k$ or a junta has $k$ relevant variables). In practice, one may only have a small sample from the target distribution $D$ or some domain knowledge. Could the authors discuss strategies or heuristics to approximate a positive distribution shift in absence of full knowledge? For example, might one use a search procedure or adaptive data collection to converge on a helpful $D’$.
- **Generality to complex domains:** How do the authors envision applying the PDS framework to more complex or structured prediction problems (images, text, etc.)? The current examples are boolean functions with fairly clear-cut “structure” that $D'$ can exploit (e.g. biases on specific bits). In more complex tasks, a positive distribution shift might correspond to something like focusing on easier sub-tasks or higher signal-to-noise data. Are there any preliminary experiments or observations on tasks beyond parity/juntas that the authors could share to illustrate PDS in action?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper appears to argue that carefully selected out-of-distribution (OOD) training data can actually improve in-distribution performance, introducing the concept of “positive distribution shift.” However, the authors state that the benefit of such shifts is “often computational rather than statistical,” which is not clearly explained. It remains unclear what specific computational advantages they refer to or how these differ from statistical improvements.

### Strengths
No strength found.

### Weaknesses
1. Clarity of presentation: It is unclear whether the section labeled “Introduction” serves as a motivation, problem formulation, or a mix of both. The narrative does not clearly define the research question or the contribution.

2. Ambiguity in notation and exposition: The description in Lines 57–74 (Page 2) is difficult to follow. The notation appears unorthodox and lacks explanation of key terms and symbols. This section is crucial for understanding the rest of the paper, yet it remains opaque.

Given that these issues make the core idea and mathematical formulation unclear, I was unable to evaluate the remainder of the paper in a meaningful way.

### Questions
I would carefully rewrite notations and claims before sending it out for further peer review.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The authors propose a novel notion of positive distribution shift (PDS) learnability and investigate its properties. PDS learnability refers to the learnability when the learner observes samples from a well-behaved distribution to construct a classifier for the test distribution. The authors first analyze PDS learnability in situations where the training distribution can be chosen depending on the underlying labeling rule, and show that any polynomial-size circuit can be PDS learnable by a feed-forward neural network with polynomial runtime. Under the non-PDS setup, even constant-depth circuits are not learnable. Subsequently, the authors elucidate cases where the training distribution depends only on the class of the underlying labeling rule instead of the labeling rule itself. They then show that noisy parity and junta problems are PDS learnable by the gradient descent algorithm with a polynomial number of iterations. The authors further demonstrate the connection between PDS learnability and non-adaptive membership query learnability.

### Strengths
The paper is well-written and grounded in solid theory. The motivation behind the PDS framework is clearly presented: to illustrate the computational advantages of using an alternative feature distribution rather than the testing feature distribution. The authors use the PDS framework to demonstrate the learnability of several tasks via practical algorithms. These learnability problems are relevant and likely to interest conference attendees.

The DS-PAC learnability of the parity and $k$-junta problems by the gradient descent algorithm with ReLU networks is both interesting and novel. These findings not only show polynomial runtime and sample complexity for the parity and $k$-junta problems but also highlight the learning capability of practical deep learning algorithms.

### Weaknesses
The practicality of the PDS framework is unclear. For a practical learner, it is not feasible to choose or construct the training distribution based on the underlying labeling rule and/or the test distribution, since both are unknown to the learner. Consequently, DS learnability should be interpreted as an oracle model, where the training distribution is selected by an oracle. To address practicality without relying on an oracle, one option is to permit the learner to observe a few samples from the test distribution when constructing the training distribution. However, this setup may reduce to the non-adaptive membership query problem. Thus, the PDS framework may primarily serve as a theoretical tool for deriving non-adaptive membership query learnability, rather than as a practical method. 

In particular, while the f-PDS setup is intriguing as a theoretical puzzle, its practical value is unclear. This definition assumes the training distribution is constructed adaptively with respect to the underlying labeling rule. In practice, a learner may need to observe labeled samples under the testing distribution to construct the training distribution. In such scenarios, it may be preferable for the learner to directly learn the classifier from the observed labeled samples under the testing distribution, rather than constructing a separate training distribution.

The statements of the theorems lack clarity. In the definitions of Defs 3.1, 4.1, and 4.2, learnability is defined as the existence of a well-behaved training distribution $D'$ for all testing distributions $D$. However, in Theorems 3.2, 4.3, and 4.6, for example, the authors also mention the existence of (another?) training distribution for all testing distributions, in addition to the learnability. This misalignment between the definitions and the statements of the theorems is confusing.

### Questions
- What concrete scenarios allow constructing a “well‑behaved” training distribution $D′$ without oracle access to the labeling rule or target distribution?
- Can the authors construct a practical procedure to synthesize $D'$?

### Soundness
2

### Presentation
3

### Contribution
3
