# On The Representation Properties Of The Perturb-Softmax And The Perturb-Argmax Probability Distributions

- Decision: Reject
- Scores: 6, 3, 5

## Abstract
The Gumbel-Softmax probability distribution allows learning discrete tokens in generative learning, whereas the Gumbel-Argmax probability distribution is useful in learning discrete structures in discriminative learning. Despite the efforts invested in optimizing these models, their properties are underexplored. In this work, we investigate their representation properties and determine for which families of parameters these probability distributions are complete, that is, can represent any probability distribution, and minimal, i.e., can represent a probability distribution uniquely. We rely on convexity and differentiability to determine these conditions and extend this framework to general probability models, denoted Perturb-Softmax and Perturb-Argmax. We conclude the analysis by identifying two sets of parameters that satisfy these assumptions and thus admit a complete and minimal representation. A faster convergence rate of Gaussian-Softmax in comparison to Gumbel-Softmax further motivates our study, as the experimental evaluation validates.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper characterizes the the ability of particular parametrized families of probability mass functions (PMFs) to fit an arbitrary PMF on a finite probability space. The main result is a set of conditions under which the Perturbed-Softmax and Perturbed-Argmax families enjoy completeness (whether the parameter-to-distribution map is onto) and/or minimality (whether the parameter-to-distribution map is one-to-one).

### Strengths
- The paper is concise and overall well-written (see Weaknesses for feedback). In particular, the proofs in the main text and footnotes are helpful and executed well; usually, this type of presentation is hard to get right.
- The mathematical analysis is elegant and general enough to account for any perturbation distribution.
- The experimental testbed is generally helpful, although I do not believe the provide much insight regarding the theory (see Weaknesses).
- The appendices are helpful for making the paper self-contained overall, but the proof appendices should not be theorem-proof lists. Try to make them self-contained as well; concretely, please reintroduce the theorem, add proof outlines, discuss any techniques that are novel to this paper versus those that are adapted from others with citations, etc)/

### Weaknesses
- The earlier part of the presentation could be improved, in that components from the Related Work and Preliminaries could be incorporated into the Introduction. For example, I don't believe the paper mentions one of the major motivations of the Gumbel-Softmax as an approximation to the discrete sampling operation in VAEs that can be backpropagated through. Similarly, while online learning methods are stated as a motivation, the reader should be able to know where the perturbed-argmax/softmax operations appear in their methods, even without having a background in online learning. Equations for the perturbed operations could appear much earlier in the text, so that the Gumbel-softmax/argmax, Gaussian-softmax/argmax, and their perturbed variants are not so abstract. 
- Unless I may be misunderstanding, their seem to be some technical errors in the main text proofs. For example, the proof of Theorem 4.1 does not use the $h_i$ function, so it must be incomplete. This assumption is in fact used in Theorem 5.1, but not included in the theorem statement.
- I do not believe the current experiments align well with rest of the paper. In particular, what is shown is that Gaussian perturbations result in a predictor that learns faster with respect to validation loss. However, the rest of the paper is about the representation capabilities of parametrized models, and if I understand correctly, the Gumbel and Gaussian perturbed models have the same representation properties. Moreover, this difference between Gaussian and Gumbel performance seems to already be understood in theory based on these concentration properties shown earlier in the paper. I felt that an experiment which included a model that did not satisfy completeness conditions (within the same setup as 6.2) and shows the bias of this model at the learned minimizer would be much more illuminating.

### Questions
- Should line 160 only consider $p \in \operatorname{ri}(\Delta)$?
- Should Theorem 3.1 include the sub-Gaussianity parameter?
- Should line 240 say "Gumbel" instead of "exponential family"?
- In Theorem 4.1, what does "whose cumulative distribution decays to zero" mean? Can you make this precise, as CDFs do not decay to zero?

### Soundness
3

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
5

### Summary
The paper studies the Gumbel-max distributions in the mathematical view. The authors especially investigate the completeness, identifiability, and minimality of the distributions, i.e., the Purturb-Maxes. Finally, the advantage of Gaussian-softmax, which is an example of the Purturb-softmax, is demonstrated in the experiment section.

### Strengths
The paper is clearly written and well-organized.

The work is theoretically grounded.

### Weaknesses
Even though this is a theoretical paper, the work requires empirical evidence more than the ones suggested in the manuscript. 

Also, it is not clear how the current experiments are related to the theories provided in the main body.

The proofs can be moved to the appendix.

### Questions
In practice, the Gumbel-softmax is more widely utilized than the Gumbel-max, and the temperature parameter is naturally added to the distribution setting. However, it seems that it is dropped in the theoretical analysis. Why is that?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper studies representation properties of "Perturb-Softmax" and "Perturb-Argmax" distributions, generalizations of the Gumbel-Softmax/Gumbel-Max distributions used in a significant line of prior machine learning work on discrete random variables. They show theorems establishing general representation results, and experiments arguing that perturbing with a Gaussian rather than a Gumbel yields improved learning performance.

### Strengths
The idea of alternatives to the Gumbel trick is interesting. It indeed relates to FTPL (and similarly, I think, to various non-exponential mechanisms used in differential privacy), but in this context to my knowledge the Gumbel version is the only one that has been thoroughly considered.

The experimental results are promising that learning with Gaussian noise may be preferable to Gumbel noise.

### Weaknesses
While the setting of this paper is interesting, the presentation is a little bit disjointed:
- Your conclusion says that "Our main contribution is a theoretical study of the representation properties of Gumbel-Softmax and Gumbel-Argmax probability models." I think this is not really true; the basic derivation of Gumbel-Argmax shows that it is obviously equivalent to general softmax parameterizations of discrete distributions, in which case the completeness and minimality properties are well-known. While I don't think either of the two Gumbel-Softmax papers made explicit arguments about completeness, both included a temperature parameter, and the equivalence of Gumbel-Softmax to Gumbel-Argmax in the limit of low temperatures – explicitly stated as Proposition 1 part (c) of Maddison et al. – immediately and obviously implies completeness there. Thus most of these results were implicit in the literature already and the new ones are not _especially_ interesting in the machine learning context.
- What is quite interesting, though, is the claim that perturbing with other forms of noise can yield the same kind of completeness (and, less relevantly for learning, minimality) properties. The experiments with Gaussian-Softmax also indicate that other kinds of noise are worth exploring, although it would be nice to have a better understanding of _why_ (see discussion about the "convergence rates" argument below) or of how much improvement can be seen in a wider variety of settings. So, I think the paper should be somewhat reframed as being about the generalized Perturb-*max models, and how other choices than Gumbel can be better.
    - To this end: while the experiments here are definitely not _nothing_, they also contain no detail other than "here is the final approximation quality," and they're also not of an especially exciting scale in 2024. Ideally, I'd like to see some better poking at why optimization worked better in a simple case, e.g. one of the fixed discrete target distributions (is the parameter vector $\\theta$ to get a good approximation simpler? if not, is something about the optimization landscape simpler?). It would also be nice to have experiments for a more modern model using this as a component; just take one of the many recent models citing these papers and swap out the Gumbel.

But, before doing that, a vital issue:

- Theorem 5.1 as stated here is **plainly false**. In the theorem statement, there is no constraint on the form of random variable $\gamma$. First, they are presumably intended to be iid, but maybe this doesn't matter. So, consider using the random variable $\gamma_i$ which is a point mass at 0. Then the Perturb-Argmax distribution is $\\mathbb{E}_\\gamma[ \\mathrm{arg\\,max}\\; \\theta + \\gamma ] = \\mathrm{arg\\,max}\\; \\theta$; depending on exactly how we define argmax's behaviour in the case of ties, this can _only_ produce one-hot distributions. Since the structure of your proof is that one-hots are in the closure of $\\mathcal P$ and that this closure is convex, presumably you didn't notice this when adapting the proof of Theorem 4.1, since indeed that first property is certainly true. Presumably, then, the claim that the closure of $\\mathcal P$ is convex is false in this case. I don't know where the flaw is, but something must be wrong. In particular, since Theorem 4.1 follows a similar structure, it is particularly concerning that there is some unknown flaw that might also apply to that theorem's proof. While it's not immediately obvious to me what the problem is, I'm not so confident in how to carefully apply Rockafellar's results that you use (which are, I think, stated for $\\mathbb R^d$) to the probability simplex, and so am not confident that it's a mistake in Theorem 5.1 specifically. I cannot see arguing for anything other than rejecting the paper unless this mistake is identified and corrected, and a full correct proof provided during the rebuttal process.

Perhaps relatedly, the mathematical discussion and definition in several aspects of this paper is quite sloppy. Here are a few reasonably important ones:
- Line 195 claims that "The fundamental theorem of extreme value statistics asserts the equivalence between the softmax distribution in Equation 4 and the Gumbel-Argmax distribution in Equation 2." I don't see how this is in any way true. You only cited a 60-page lecture series which does not use this phrase to identify a theorem; presumably, you mean a [version of this theorem](https://en.wikipedia.org/wiki/Fisher%E2%80%93Tippett%E2%80%93Gnedenko_theorem), which states that the maximum of $n$ iid values can have one three asymptotic behaviours as $n \\to \\infty$, one of which is the Gumbel distribution. I don't see at all what this has to do with the distribution of a _soft_ max of a finite number of values. Luckily, though, you don't seem to use this seemingly-incorrect claim again.
- Section 3.4 "Convergence rates" compares the upper bounds on concentration of Lipschitz functions of a Gaussian variable and a Gumbel variable, and since the former is faster than the latter, vaguely imply that this means that Gaussian-Softmax is better than Gumbel-Softmax. But the purpose of the remainder of your paper is that you would use the two to identify _the exact same discrete distribution_, and thus estimating $\mathbb E\_{\\gamma \\sim g}[ f(\\theta, x, \\gamma) ]$ by Monte Carlo samples of $\\gamma$ will be _identical_ between equivalent choices of $\\theta$ with different distributions for $\\gamma$. These differences in upper bounds (which are worst-case over functions $\\gamma \\mapsto f(\\theta, x, \\gamma)$) are not relevant here.

And here are a few examples that are easy to fix, but indicative that perhaps you were not as careful in writing as you should have been:
- On line 172 you say that the probability density function of the Gumbel is $\\hat g(t) = e^{-e^{-(t+c)}}$ where $c \\approx 0.5772$. This is not true; this is the _cumulative_ density function, specifically of a Gumbel with mean $c$ and scale $1$. While scale $1$ is standard for the Gumbel-max trick, there is absolutely no reason to use mean $c$; it doesn't break anything, but by far the more typical choice would be to use mean $0$.
- Theorem 4.1 says "let $\gamma = (\\gamma\_1, \dots, \\gamma\_d)$ be a vector of random variables whose cumulative distribution decays to zero as $\\gamma$ approaches $\\pm \infty$." First, presumably you mean that the $\\gamma_i$ are iid, and it is the distribution of each of those that has the appropriate decay property. Secondly: it is trivially true of any real-valued random variable that its cumulative distribution function decays to zero as you approach $-\\infty$, and to _one_ as you approach $\\infty$. Perhaps you meant that the measure of sets $(-\\infty, -t)$ and $(t, \\infty)$ should approach zero as $t \\to \\infty$? But this is again automatically true of any real-valued variable. I actually have no idea what you mean here.
- Line 719: "A multivariate function is differentiable if its directional derivative is the same in every direction $v \\in \\mathbb R^d$, namely $\\nabla f(\\theta) = \\nabla_v f(\\theta)$ for every $v \\in \\mathbb R^d$." This is not at all true, as should be clear from the fact that your equation is equating a vector to a scalar. Rather, the directional derivatives should be _consistent_, i.e. $\\nabla f(\\theta) \cdot v = \\nabla_v f(\\theta)$ for all $v$.

Minor points of terminology:
- Your definition of identifiability is different than the one I've always heard before, and is also used e.g. [by Wikipedia](https://en.wikipedia.org/wiki/Identifiability), which is exactly what you call "minimal." I don't see much reason in giving a special name to the bijection case here, and particularly in giving one that is so widely used to mean something slightly different; I would strongly encourage you to change the terminology to "complete" (or perhaps "universal" would be more common in learning contexts) when the map from parameters to probability distributions is (almost) a surjection, and "identifiable" when it is an injection.
- Typically one would put `\appendix` at the start of the appendix, which would name the appendix A rather than 8.
- Line 713: "Convexity is a one-dimensional property." This is a strange statement; while you can define it in one dimension and extend it to other cases as you do here, there are also (many) direct definitions of convexity that work for any input vector space.

Trivial typos, etc that I happened to notice:
- line 160: you wrote $softmax$ instead of $\\mathrm{softmax}$
- "subsetset" on line 209
- Footnote 3: "dominant convergence theorem" should be "dominated convergence theorem"

### Questions
- How can Theorem 5.1 be corrected? Does the mistake also apply to Theorem 4.1?

- Do you have any insights as to why Gaussian-Softmax seems to be more learnable in these settings than Gumbel-Softmax? Can those insights be generalized to help guide to other noise distributions that might be even better?

### Soundness
2

### Presentation
2

### Contribution
2
