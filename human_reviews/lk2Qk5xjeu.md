# Unifying Causal Representation Learning with the Invariance Principle

- Avg Score: 7.00
- Decision: Accept (Poster)
- Scores: 8, 6, 8, 6

## Abstract
Causal representation learning (CRL) aims at recovering latent causal variables from high-dimensional observations to solve causal downstream tasks, such as predicting the effect of new interventions or more robust classification. 
  A plethora of methods have been developed, each tackling carefully crafted problem settings that lead to different types of identifiability. 
  These different settings are widely assumed to be important because they are often linked to different rungs of Pearl's causal hierarchy, even though this correspondence is not always exact.
    This work shows that instead of strictly conforming to this hierarchical mapping, *many causal representation learning approaches methodologically align their representations with inherent data symmetries.*
  Identification of causal variables is guided by invariance principles that are not necessarily causal. 
  This result allows us to unify many existing approaches in a single method that can mix and match different assumptions, including non-causal ones, based on the invariance relevant to the problem at hand. 
  It also significantly benefits applicability, which we demonstrate by improving treatment effect estimation on real-world high-dimensional ecological data. Overall, this paper clarifies the role of causal assumptions in the discovery of causal variables and shifts the focus to preserving data symmetries.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper presents a unified framework for causal representation learning by using invariance principles to identify latent variables across various settings. The proposed framework unifies various existing CRL approaches, through the invariance principles. It introduces theoretical guarantees for latent variable identifiability, and demonstrates improved treatment effect estimation using synthetic experiments and real-world causal benchmarks.

### Strengths
- The paper is very well-written, with intuition provided for the definitions and main results.

- The review of related works is thorough.

- This work provides a valuable perspective by unifying multiple CRL approaches under a general framework and provides a novel perspective for the causal representation learning.

- The formalization of identifiability definitions and theorems is clear and sound.

### Weaknesses
While the paper presents a strong theoretical foundation, the framework is tested on limited data types, which may not reflect its applicability across various CRL settings.

### Questions
In practice, how do you choose the hyperparameters such as $\lambda$?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Many approaches have emerged in recent years that tackle different variants of the causal representation learning (CRL) problem, under different assumptions and using different types of algorithms, and proving different levels of identifiability. This paper presents a unifying framework that allows the commonalities and differences between these different approaches to be more easily recognized.

### Strengths
In the quickly growing body of literature on CRL, it is becoming hard to see the forest for the trees. In this paper, the different main types are clearly delineated and discussed. I think this is an extremely valuable contribution to the field of CRL.

### Weaknesses
- Section 2 was very hard to read, and the definitions introduced here turn out to be misformulated. I only started to understand the intended meaning of definition 2.1 after reading parts of section D in the appendix: In multiview CRL scenarios (D.1), indeed the invariance property says that the two random variables take identical values. This type of invariance is captured by definition 2.1. But in multi-environment CRL (D.2), the invariance properties are not a function of the random variables (or their realizations), but of their density functions. The definition of $\iota$ is inappropriate in the latter case (specifically, one can't write $\iota(\mathbf{a})$ or $\iota:\mathbb{R}^{\lvert A \rvert} \to \mathcal{M}$ if $\iota$ is meant to be a function of the *density* of $\mathbf{a}$). Since the core contribution of this paper is to unify these different notions of invariance under a single theory (with a single notation), I believe it is of paramount importance that this notation is clear and correct.
- The following is not a weakness of the paper, but a weakness of the paper-venue pairing: A proper exposition of the theory in section 2 would require some examples to be presented, that demonstrate how the new definitions play out in known scenarios. At present, this technical section is very difficult to get through because there is no room for such examples, or for any redundancy to help the reader confirm they are on the right track. If the examples in appendix D could be integrated into the main text, the result would be a much better paper. Another sign this paper wants more space is that this paper has more to say than fits in 10 pages, is that of the four main contributions listed on page 2, only the fourth and part of the first are in the main text. Indeed I think the most valuable sections of this paper are currently in the appendix. Presenting this work as-is might be doing it a disservice in the long run.
- There were some more flaws in the theory on page 6, but I believe these are easily fixed. Proposition 3.3 says "only if", but what the proof shows is "if" (i.e., if (i) and (ii) are met, then the identifiability is established). Now the statements of propositions 3.2 and 3.3 are contradictory. The statement of proposition 3.2 should make clear that it only shows that identification is not possible *in general* here.

### Questions
- What is the content of assumption 2.1? It looks like this just defines $\mathbf{x}_{V_i}$, but oddly this definition doesn't involve $V_i$ anywhere, so it remains unclear what it means to say that $V_i$ is "unique known". I assume there is an implicit role for $V_i$ here? (As an aside, wouldn't the notation $\mathbf{x}^{V_i}$ be more consistent with the use of superscripts on the bottom of page 3?) Additionally, the "in particular" in the last sentence suggests that this is implied by the previous. If that's indeed the intended meaning, could you explain how this follows?
#### Small comments
- Please, sort the references alphabetically!
- $\iota$ is interchangeably called "invariance" and "invariant" property.
- line 215: "A subset of latent variable $\mathbf{z}_A$" - "variable" should be "variables". But then it sounds like a subset of $\mathbf{z}_A$ is meant, while $\mathbf{z}_A$ is the subset. I suggest to move "of the latent variables" to after the math.
- Definition 3.3: Replace "s.t." by "where" ("s.t." is used for conditions).
- page 22: I think on line 1145, "interior" should just be "subset". "equity" should be "equality", and on line 1152, in the big intersection the $S_k$ is missing.
#### Language-level comments
- l126: "smooth function**s**"
- l133: "violates **the** invariance principle"
- l146: "present**s**"
- l157: "interio**r**"
- the image of a function is sometimes denoted Im (l162), sometimes $\mathcal{I}$ (l951)
- l180: "as a": remove "as"
- l363: "Mult**i**-task"
- l987: remove "is a"
- l1109: "remain**s**"

### Soundness
2

### Presentation
2

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors provide a general framework for identifying a group of $N$ random variables that encode invariances between a set of $K$ joint probability distributions, each over $D$ variables. Their framework describes many existing causal representation learning (CRL) algorithms and their corresponding (partial) identifiability results. The authors present their insights as methodologically connecting CRL across different settings where invariances may correspond to, for instance, the effect of specific interventions over a shared latent (natural) structural causal model, or various data missingness regimes.

### Strengths
The authors motivate their approach well by describing the breadth of assumptions made across different CRL settings. Their framework is laid out formally, and the boxed plain-English commentary is helpful for first-time reading.

### Weaknesses
(Section 4) As the authors aim to unify many existing problem settings, discussion on the application of the authors' framework is broad. In the main text this discussion is often informal. The authors could explicitly compare the invariance objectives for between common CRL approaches/settings in the main text. Does this lead to any surprising insights, for instance by showing methodological similarities between settings for which no comparison had previously been discussed?

(Future work) To my understanding, the variables preserving symmetries across different data pockets are not necessarily causal, but there are many existing problem settings for which certain corresponding symmetries and their interpretations can lead to causal variables being (block) identified. Do the authors see potential for their framework to be used to discover now CRL settings/approaches.

### Questions
(Future work) The authors' sufficiency constraint (Constraint 3.2) in general requires a supervised approach to enforce since the ground truth invariant representation $\mathbf{z}_{A_i}$ is unknown. Does this provide insights into the impossibility of unsupervised/semi supervised CRL in certain settings?

(Line 79) Is there a reason that the authors' framework is limited to describing nonparametric CRL? Are there known cases in which parametric assumptions cannot be represented through data invariances? Perhaps some parametric assumptions allow for CRL in a setting with only one dataset?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work provides a unifying theory for multiple causal representation learning works by using the fact that all these works are learning some form of invariance. It also acts as a review work that neatly characterises the different problem settings in causal representation learning.

### Strengths
- The paper is comprehensive and the unifying theory seems like a very good way to consolidate the findings in the causal representation learning literature.

### Weaknesses
- I think some of the contributions are over stated, contribution 2 and 3 in L86 and L90 are already known. While it's great that these are included, I would not put them down as contributions of the work. It may mislead the reader to think that there are new findings here. The other contributions are enough for a good paper.
- This work can be *greatly* improved if the grey boxes after the definitions, assumptions etc. in sections 2 and 3 included explicit examples relating to literature. This is not present in section 4 and only partially in section D. With this I mean something like, what is the invariance property in multi environment learning and temporal learning? What is the space $\mathcal{M}$ in all cases? The intuition in L135 to L143 is ok, but I would like to see *explicit* definitions in each related work case to be thoroughly convinced that each case is covered by the theory presented. The same is true of things like definition 2.2. Explicit examples of why the definition can cover multiview, multienvironment etc. will help greatly. I'm not sure saying the data is "paired" in L170 is enough. In short, I would like to see how each related work fits in mathematically with your theory. I'm also not convinced the tables at the end do this.
- I'm very unsure how the theory lead to a method that can work on the ISTAnt dataset as mentioned in L54. I can't see the reason why previous assumptions were insufficient written anywhere, or why the theory directly leads to the application of the domain generalisation method? As this is stated as a main contribution, this needs to be a lot clearer than it is. 

I think this work will be of great interest to the community, but more careful exposition of why most CRL works can be seen as a special case, and more careful statement of the contributions, would greatly help the clarity of this work.

### Questions
- L258: Where is H defined? If H is used in constraint 3.2, it should be defined, or the definition pointed to.
- L63: What is a data pocket? This is used a few times in the work but has not been defined.

### Soundness
2

### Presentation
2

### Contribution
4
