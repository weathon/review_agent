# Towards Predicate-powered Learning

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 3, 3

## Abstract
The traditional approach to data-driven learning has become increasingly demanding in terms of its training data and computational resources. This work further develops a new paradigm of learning using predicates to reduce the need of data in learning. Among many recent efforts towards the same direction, learning using statistical invariants (LUSI) has been proposed to be the new paradigm of learning. Building on top of LUSI and to break the ``brute force'' learning trend, we build towards a generalized theory of predicates and the invariants. The primary objective of this work is to propose an Extended Structure Risk Minimization (ESRM) paradigm with predicates, and provide a theoretical justification of the need for predicates in learning problems from both data complexity and model complexity perspectives. In this work, we show that predicates not only can aid in reducing the need for data in training, but they are also imperative for a highly efficient model. 

Our primary contributions consist of the following: I) Proposing an extension to the structure risk minimization paradigm of learning, and II) Proving the efficacy of predicates in reducing both the data complexity and the model complexity.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In this paper, the authors present an extension to the structure risk minimization.
Here the authors present proofs for the impact of effective predicates on learning for data and model complexity.
The authors then provide an empirical evaluation for direct and statistical invariants.

### Strengths
Starting from effective predicates, the authors then build proofs showing a reduction in data and model complexity.

### Weaknesses
From the paper, it does not seem like finding effective predicates is easy, nor integrating them into the learning algorithm. 

The empirical examination does not properly support the claims of the paper, nor serve as a good example of the proofs working and this is the weakest part of the paper.

### Questions
Does it make sense and invert the problem, i.e., ask, what data points are needed for learning based on the predicates? This could be useful for use cases where exploration is feasible.

### Soundness
3 good

### Presentation
3 good

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
This paper may have a good idea, but is not ready for publication. There are too many assumptions made by the authors that are not made explicit. (Caveat: I am not a researcher in the area of the paper, but I should be able to understand it). The experiments are not up to the standard of a top conference. They seem to show that the proposed idea does not work.

### Strengths
see above

### Weaknesses
First, I think what you call predicates should be called meta-predicates, as they are predicates about the data, not predicates in the data (e.g., as you might find in relational data). I had assumed that you meant predicate as in logic that is a Boolean function, rather than as in the predicate of a sentence. In any case you need to tell us what a predicate is.

There are lots of problems with notation:
In pseudocode 1 (and surrounding text). Are the w_i are probabilities? Are the hypotheses disjoint? But then in theorem 1, the weight update is assume to be positive; surely it can't be positive for all hypotheses. (I think the problem is that there is an explicit quantification that is not given. For it mean "for all n" or "there exists n"? - as I don't see what n is). In the equation after equation (1), \Delta w_i is always positive; w_i needs to be explained better,

You are assuming much more of the reader than can be assumed. You tell us that <f,\phi> is a pair. And then a very strange sentence in bold; I'm not sure why we would assume any interaction between f and \phi; it is just a pair.  I think you mean a function of f and \phi with some properties that you don't state. It is then fine to use the inner product as example.

In equation (2) why is it only "approximately equal"? Surely it should be =

Please don't say "which resembles human learning" without saying how it resembles human learning and providing evidence (eg, a reference) that says that human learning is like that.

On the bottom of page 5, how can there not be "countable many data points"? (I'm having trouble with "without much loss of generality; I'm thinking of one of the dimensions is time, and we want to predict the future from that past; I'm guessing that is a case that is not covered ;^)

On the top of page 6, "Naturally, if we have..." seems to imply there is no noise in the data. If that is an assumption it should be made explicit.

On definition 2, please don't use epsilon for two different concepts -- I see now that the epsilon in expsilon-ball is a different font than the < epsilon  (they have to be different as they different units), but it was very confusing. This should be called an equivalent relation as it is not transitive.

Definition 3 doesn't make sense. I don't think you want "Let".

Definition 5 doesn't make sense. What is "they"? I can't see a definition of sub-cover here. Definition 6 seems to also make a claim "Then, there is a..." which isn't obvious.

Why is the union not unique? I don't think I understand what it is then.

Page 8, at the bottom "same prediction" seems to imply that y is discrete. If that is the case, it should be made explicit. (Or what does "same" mean; do you want some epsilon? ) 

The experiments are not up to the standards of a top conference. The experiments are the average of 10 or 5 runs. No error bars are given, but I would expect the variance to be so high that the results are meaningless. (All the numbers in table 1 look the same to me).

### Questions
There is no explicit definition of a predicate. I think it is a function on examples. But isn't that a property? What makes these properties special in that they can be treated differently from the other properties in the data?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper develops a theory to prove that predicate-powered learning reduces data complexity to build models (not many instances are needed for training) and reduces model complexity.

### Strengths
- Attempt to formalize the use of predicates on learning tasks
- Attempt to prove data and model complexity reduction

### Weaknesses
- Lack of background on some of the concepts used
- Domain knowledge is hard to code and your work depend on domain knowledge
- Only one experiment with one dataset.

### Questions
I found this paper very hard to read maybe because the concept of predicate used in this paper does not match the concept of logic predicate (knowledge) I have in mind. The title of this paper mentions the use of predicates. I could not find any predicate example and how they are used/encoded in this work. It'd be ideal to give an example. Authors mention mean and variance, but these can be well used for numerical data. What about categorical data, where predicates are actually useful?

Theorem 1: talks about experditing the learning rate by providing useful predicates, however, the proof mentions number of samples required for learning of SRM being smaller than for ESRM. What is the relation between the reduction in number of samples and learning rate (smaller data?) and how about quality of the model with the reduction in data?

In Pseudocode 1, line 4, what are the predicates? How do you obtain them?

S2.2: ESRM rule proposed earlier: are you talking about Pseudocode 1? Is it a rule? Or are you talking about some rule used in the original SRM?

There are some works that embed expert knowledge to learn new models (e.g., https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4525246/ or https://www.jmlr.org/papers/volume17/15-444/15-444.pdf). Knowledge can be embedded through propositionalization or using (first-order) relational models. I guess one of your contribs is to formalize the way knowledge is embedded. However, it looks like your method is limited to numerical or image data.

Typos etc:
to differentiate the among those hypothesis classes. --> to differentiate among those hypothesis classes.

a structure risk minimization rule that we allow the weights -->  a structure risk minimization rule that allows (will allow?) the weights

hat the model should predicate the same --> hat the model should predict (??) the same

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper considers extending work of Vapnik on learning using statistical invariants. Particularly, they propose an extension of structural risk minimization in which the weights $w_i$ associated to classes $H_i$ are updated to favor those $H_i$ that are closer to satisfying desired invariants. They also try to formalize how predicates can divide the data into equivalence classes, leading to reduced data and model complexity requirements.

### Strengths
The idea of adjusting weights in SRM to reflect adherence to invariants offers an interesting and straightforward approach for integrating prior knowledge. And it seems worthwhile to explore quantifying how predicates can reduce the need for additional data.

### Weaknesses
Various typos (e.g. p.2 “the ability to differentiate the among those hypothesis classes”, p.3 “control the modeling behavior more granular”) and nonstandard terminology (e.g. "structure risk minimization", "equivalent relation") make the paper hard to read.


The weight update scheme for ESRM is not rigorously developed:

-The pseudocode for ESRM is hard to follow: Perhaps the intention was for $\epsilon_i$ to be the minimum of the set rather than the set itself. Since $h$ may belong to multiple $H_i$, the objective is also ambiguous as written.

-Theorem 1 seems to only handwave that if we are able to update SRM weights to favor the class $H_i$ containing an optimal model, then we expect the performance of SRM to improve. The subsequent section does not adequately clarify how this would be achieved.

Section 2.3 is hard to contextualize with the rest of the paper:

-The proposed aim is to connect statistical invariants as defined by Vapnik to the concept of invariants elsewhere (e.g. models that require the same output on the orbit of $x$ under some group action). How the analysis achieves this is fuzzy, perhaps due to a lack of definitions. Might be nice to look at the recent work https://proceedings.mlr.press/v128/vapnik20a.html, which has more examples of predicates describing symmetries.

As written, the definitions and proofs in section 4 and 5 are not rigorous or correct. For example, in Definition 1 the meaning of $P(y_k | \mathcal{B})$ is not clear, perhaps intended to be an expected value over $\mathcal{B}$. It is also confusing that the definition seems to be trivially satisfied by setting $\tilde{D}=\mathcal{B}$. This confusion is exacerbated in Definition 2, where the meaning of $P(y_k | e_i; \phi)$ is ambiguous. The proof of Theorem 2 is not rigorous, affecting the proof of Theorem 3. Similar problems with Proposition 1.

The experiments in section 6 would be improved by graphics comparing the baseline to the invariant augmented models, and employing some form of uncertainty quantification. The connection to ESRM could also be better developed.

### Questions
-In section 2.2, it is suggested to update weights for $H_i$ proportional to $|\sum f(x_i) \phi(x_i) - y_i \phi(x_i)|$. Is $f$ an arbitrary element of $H_i$? The purpose of subsequently decomposing $f$ and $\phi$ is also unclear.

-In section 2.3, could you clarify the definition and domain of the linear functional $L_f$? It reads as if $L_f$ might map $\phi$ to composition $f \circ \phi$, but it is not clear to me why this would be linear.

-In section 4, what precisely are $P(y_k |\mathcal{B})$ and $P(y_k|e_i; \phi )$? Why do you assume the difference is less than $\varepsilon$ for all $\varepsilon$, rather than equal to zero? 

-In the proof of Proposition 1, why do you require the axiom of choice when we're dealing with finite samples?

### Soundness
1 poor

### Presentation
2 fair

### Contribution
2 fair
