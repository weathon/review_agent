# Event-Level Causality: A Framework to Unify Causal modeling and Explainable AI

- Avg Score: 4.40
- Decision: Reject
- Scores: 4, 6, 6, 2, 4

## Abstract
Causal relations are typically modeled between random variables (RVs), yet in real-world settings, it is events that cause other events, not RVs causing RVs. We formalize this perspective as Event-Level Causality (ELC), under which Bayesian network structure learning and many Explainable AI (XAI) methods can be viewed as special cases. ELC increases the flexibility of causal modeling by capturing dependencies beyond classical structure learning. It also strengthens XAI by rigorously linking feature importance to causality and showing that different XAI models approximate a principled objective function with varying degrees of fidelity. We propose a new approximation of this objective that, in experiments, clearly outperforms benchmarks such as LIME, L2X, SHAP, and INVASE.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Event-Level Causality (ELC), a framework proposing that causal relationships occur between specific events (realizations of random variables) rather than solely between the random variables themselves. ELC generalizes traditional Bayesian Structure Learning by allowing parent sets ('parenthood functions') to vary depending on the specific realization of preceding variables. The authors argue ELC provides a more natural view of causality and bridges causal modeling with XAI, particularly instance-wise feature selection methods. Under ELC, many XAI methods approximate a principled objective for identifying the minimal set of features rendering an output conditionally independent of others given an instance. A new XAI method, Hide&Seek, is proposed based on this objective, designed to avoid information leakage issues present in some prior methods (e.g., INVASE).

### Strengths
1. The core idea of formalizing causality at the event level is thought-provoking and offers a potentially richer, more flexible alternative to standard RV-level causal graphs, especially for capturing context-specific dependencies. 

2. Hide&Seek demonstrates superior performance compared to established XAI benchmarks on synthetic tasks designed to test global, instance-wise, and context-specific feature importance recovery.

### Weaknesses
1. The paper relaxes the Causal Markov Assumption but retains Causal Sufficiency, Faithfulness, and Known Order. The practical implications and limitations of relying on these assumptions in the context of event-level variability need further discussion, especially regarding identifiability guarantees beyond simple cases. 

2. The scalability of learning a full ELC structure (beyond the single-node XAI setting) is not explored. 

3. The relationship between ELC and existing work on context-specific independence (CSI), labeled DAGs, and tree-CPDs could be elaborated further.

### Questions
1. Under what precise conditions can the optimal event-level parenthood functions $\mathcal{S}^{*}(x)$ be uniquely identified from observational data, given the known order assumption but allowing for event-level graph changes? How does this compare to identifiability in standard BSL? 

2. How well does the proposed Hide&Seek loss (Eq. 11), using fake draws, approximate the theoretical ELC objective (Eq. 9)? Could the discrepancy impact the identification of the truly minimal parent sets in complex dependency scenarios?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a framework for event-level structure learning and causal modeling. By relaxing the traditional causal Markov assumption, it allows each event to correspond to a potentially different DAG structure. This framework unifies traditional random variable-level causal discovery with instance-level feature importance interpretation methods. The authors further propose a specific XAI model called Hide&Seek, which effectively avoids information leakage through a dual-network architecture and feature replacement mechanism. Experiments show that Hide&Seek outperforms existing baseline methods in both feature selection accuracy and the ability to identify "switched features" on synthetic data and the MNIST image recognition task.

### Strengths
The ELC framework, first proposed, breaks down the barriers between causal modeling and XAI from an "event-level" perspective. It unifies BSL and most XAI methods as special cases, resolving the core issue of incompatible underlying assumptions between the two fields and providing a foundational theoretical framework for subsequent cross-disciplinary research.

By replacing the traditional fixed DAG assumption with the "event-level causal Markov assumption," the set of parent nodes can dynamically adjust with events. This successfully captures dynamic causal dependencies that traditional RV-level modeling cannot, better aligning with the reality of "causal relationships changing with the scenario." The method is ingeniously designed. The Hide & Seek model effectively avoids information leakage through feature replacement, boasts a simple structure, and is end-to-end differentiable.

The method is well-validated experimentally, with a systematic evaluation conducted on various synthetic data sets and MNIST. The method surpasses seven baseline models in key metrics and training efficiency, yielding compelling empirical results.

The theoretical analysis of XAI methods is in-depth, and the appendix formally analyzes the information leakage issues of existing XAI methods. This paper provides a canonical definition of XAI "feature importance" based on causal relationships, which makes up for the defects of existing XAI methods in which "feature importance" is mostly implicitly defined and lacks causal anchoring, and upgrades XAI explanations from "statistical correlation" to "causally explainable".

### Weaknesses
The paper explicitly relies on the "known topological order of variables" to distinguish causal directions, but fails to discuss solutions when this assumption breaks. This limits the applicability of the ELC framework to complex real-world problems.

Generalization to complex data scenarios is not fully verified: The MNIST experiments focus only on two simple classification tasks, "3 vs. 8," and fail to verify the model's performance on multi-class (such as 10-class MNIST), high-dimensional data (such as natural language processing and medical imaging), or data with noise or outliers, raising questions about its generalization ability.

The paper fails to discuss the applicability of Hide&Seek to non-classification tasks, indicating insufficient task coverage. The paper uses the ε interval approximation for continuous variables, which avoids measure-theoretic issues but may introduce approximation errors.

A lack of comparative analysis with "counterfactual XAI": The paper briefly distinguishes the "forward/backward" differences between ELC and counterfactual reasoning, but fails to compare their advantages and disadvantages in practical application metrics such as "explanation credibility" and "user acceptance," nor does it explore the possibility of combining ELC with counterfactual methods. Although Appendix B attempts to distinguish between event-level causality and counterfactual reasoning, the boundaries and complementarity between the two in practical applications could be further clarified.

### Questions
How can we extend the event-level causal framework when no topological order is known? Can we consider partial order or use intervening data?

Does Hide&Seek work with high-dimensional or structured data (e.g., images or text)? Does it need to be combined with convolutional or attention mechanisms? How interpretable are event-level causal graphs in real-world systems? Might they be difficult to understand due to their complex structure?

Can event-level causality be combined with counterfactual reasoning? For example, given a known SCM, can event-level parent node sets be used for more accurate counterfactual inference?

### Soundness
3

### Presentation
3

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
This paper proposes an ambitious and conceptually rich framework called Event-Level Causality (ELC), which seeks to bridge the gap between causal structure learning and explainable AI (XAI). The central argument is that traditional Bayesian structure learning (BSL) models dependencies between random variables, while real-world causation operates between events. The authors formalize this idea into an event-level causal model and propose a practical instantiation, Hide&Seek, a differentiable XAI model that approximates causal feature importance. The paper is theoretically sound, well-motivated, and relevant to both the causal inference and explainability communities. It demonstrates a deep understanding of the limitations of current causal and interpretability paradigms and offers a unifying framework that could inspire a new direction of research.

### Strengths
1. The formal relaxation of the Causal Markov Assumption to the event-level causal Markov assumption is novel and well-argued.
2. The framework elegantly unites Bayesian structure learning and instance-wise XAI under a single formalism.
3. Definitions (e.g., event-specific conditional independence) and derivations are mathematically precise and insightful.
4. The proposed Hide&Seek model is well-designed, end-to-end differentiable, and empirically validated.

### Weaknesses
1. The model assumes a known topological order among variables. This assumption is strong and unrealistic for most real-world problems where causal direction is unknown. The framework would benefit from a discussion of how to relax or learn such orderings.
2. Experiments rely primarily on synthetic data and MNIST, both low-dimensional and noise-free contexts. The model’s scalability to high-dimensional or tabular real-world datasets (e.g., genomics or finance) is untested.
3. While theoretically elegant, event-level causality may be difficult to operationalize or interpret in practice. The causal semantics of varying DAGs per event may not align with standard structural causal models (SCMs).
4. The paper positions ELC as a bridge to causal modeling, but it does not evaluate causal metrics (e.g., recovering causal parents or do-interventions). This leaves uncertainty as to whether the learned relationships are truly causal or just predictive.

### Questions
1. How can the known ordering assumption be relaxed? Could the framework integrate structure-learning techniques (e.g., NOTEARS) to infer orderings?
2. Does event-level causality preserve causal identifiability under interventions?
3. Could the event-level parenthood function be interpreted as a probabilistic distribution over potential causal structures?
4. How robust is Hide&Seek to correlated features or spurious dependencies?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper defines a notion of event-level causality, presents a method for learning it, and then uses it as a tool for explaining the outputs of AI systems. It compares this novel tool (that builds on an existing one) to existing causal feature importance tools on several use-cases, showing that it performs better.

### Strengths
The general idea of using causal models to improve local explanations of AI systems is a good one, and this paper aims to contribute to that idea. 

The definition of event-level parenthood (if worked out properly) is potentially an interesting concept that deserves its place in the causal literature.

### Weaknesses
There are several major problems with the paper. 

__A__: The formal details are not worked out well at all, and lead me to believe that they are either inconsistent, or far too convoluted. 

__B__: The formal definitions are supposed to be motivated by the fact that we can use them for explaining an output Y of an AI method. However, the setting of section 4 is much more restricted than that of section 3, to such an extent that their novel definitions are entirely unnecessary.

__C__: Contrary to what the authors claim, their notion of event-level causality is not substantially different from actual causation. 

__D__: Related to the above, there exists work that is extremely similar to the current approach, and yet it not discussed at all. 

I will make these problems more concrete in the Questions section below. 

__Some minor issues:__

__E__: There exists relevant prior work that could be mentioned. The idea of allowing for different causal structures for different realizations of variables is not novel. Firstly, when restricted to realizations of exogenous variables (or root variables), this idea appears in several works on the logic of causal models, see for example Halpern (2016) "Actual Causality", chapter 2. Secondly, the idea of having event-level parents was introduced already in Beckers (2021) "Equivalent causal models". (There the focus was exclusively on the logical setting, but the probabilistic setting can trivially be added on top of this.) 

__F__: I find the example badly chosen. Time does not cause humidity, time does not cause anything at all. So it’s strange to take T as a causal variable. It would be much more natural to just use time-indexed variables for S and L.

### Questions
__A: Questions and comments regarding the formal details.__

__Issue 1:__

Although never made explicit, the authors assume that there is a fixed total ordering of the variables, i.e., an ordering that does _not_ depend on the realization. (Without such an ordering, their definitions are not consistent.) This should be made explicit.

It is not at all clear what the authors are trying to do with their definitions. 

Here is the first interpretation of what they are trying to do. There is the standard, variable level causal Markov condition, with the accompanying joint distribution P, and the variable level parents. For any such distribution P, we can consider for any X, the variable level parents PaX, and any realization $PaX=pax$, the distribution $P(X | PaX= pax)$. In some cases, it holds that  $P(X | PaX= pax)=P(X | PaX^* = {pax*})$  for some $PaX^* \subset PaX$ and with $pax*$ the restriction of $pax$ to $PaX*$. In such a case, we can say that $PaX*$ form a set of event-level parents of X for the realization $PaX= pax$. By extension, it forms a set of event-level parents of X for any realization $X_<=x_<$ that extends $PaX= pax$ to a realization over all variables with lower index than X. 

Note that there may exist multiple choices $PaX^*$ that are subset-minimal for which this is true. (For example, if $P(Y | A=1,B=1)=P(Y | A=1)=P(Y | B=1)$, then both $A$ and $B$ form a set of event-level parents of Y.) Because of this lack of unicity, the original DAG and distribution P does not suffice to define a function $H(x_{<i})$ that returns the event-level parents of $X_i$. And _that_ is why the authors introduce what is essentially an oracle function $H$ that results in a specific choice of event-level parents for each realization. The event-level DAGs are then nothing but the DAGs that are constructed along the ordering given by H. 

The above interpretation is not in any way explicitly given in the paper. Rather, the paper presents something that seems much more general and much more complicated. This suggests that there is a second interpretation, which agrees with some parts of the first interpretation, but not all. I fail to see how this second interpretation works. We have a joint distribution P(X). Either this distribution allows for a variable-level Markov factorization, or it doesn't. Given that the authors assume that there is a fixed ordering of the variables, and given that all of the event-level Markov factorizations respect this ordering, we can use this ordering to give us a variable-level Markov factorization. But then it is wrong to present their assumption as being less restrictive, since it also relies on assuming a variable-level Markov factorization.

Rather, their framework is one which adds a further function H to the standard framework, that gives us a different Markov factorization for each realization. But this factorization has to respect the original, variable-level, factorization, doesn't it? So aren't we then left merely with H being allowed to select some subset of the original, variable-level parents, for which P tells us that these parents suffice? But then we are back at the first interpretation... Perhaps I am overlooking something, but then the authors should make this all much more clear, and give examples.

Let us assume that the first interpretation is correct. Then why not present their approach in the manner that I did? Surely that is much easier than the presentation given in the paper. Also, given the lack of unicity, it seems as if all of the hard work is being done by an oracle H. But then this overlooks the question as to what determines H, and that is the question we need to answer: what were the variables that explain some output variable? Perhaps the following issue holds the key to answering this?

__Issue 2:__

First H is explained, and then the authors switch to finding H*. I didn’t understand what to make of this. Is the idea that we are _constructing_ some H*? Or are we approximating some ground-truth H? 

__B: formal definitions from Section 3 are not necessary.__

In section 4 the authors move to the application of their definitions. But the application doesn’t seem to require their complicated definitions: we fix the output variable Y, and it is the variable with highest index, so it appears that all of the information regarding event-level parents for all others variables is irrelevant. Rather, for a given realization X=x, we are simply looking for some small subset X*=x* that explains Y=y. The fixed ordering is entirely irrelevant, and the event-level DAGs are mostly entirely irrelevant as well: all we need, is to decide which members of X were actually relevant for Y. So the novelty of their approach disappears.

__C: Actual causality__

Relatedly, contrary to what the authors claim, in this restricted XAI setting their notion of event-level causality is the same as the notion of actual causality. The authors say it is different because actual causation is backwardlooking and their approach is not. This difference disappears entirely given that the only forwardlooking that occurs here is relative to Y, the final variable. In other words, the only candidate difference is that they focus on explaining the variable Y instead of a specific event Y=y. But even that small difference does not exist, for their method focusses on actual, specific, outcomes Y=y, for that is the only way they can compare to all the other methods, which do this. In other words, they compare to local XAI methods, and that which is to be explained is some Y=y, not the variable Y. 

__D: closely related work__

They are not the first to invoke causal explanations for the purposes of local XAI, and they are not the first to compare these to other methods. In fact, given the above, their method is extremely similar to the one that is theoretically developed in Chockler and Halpern (2024) “Explaining Image Classifiers” and then implemented in several follow-up papers, the most relevant one being Chockler et al. (2025) “Causal explanations for image classifiers”. Furthermore, these papers prove that Halpern's definition of actual causality in this setting simplifies drastically, and seems to correspond pretty much to what the authors propose. (Similar results are to be found in Beckers (2022) "Causal explanations and XAI".)  So the authors should discuss this alternative approach, and explain what is novel about theirs. 

__Some minor formal issues:__

193: Why speak of H* as a structure, when it is a function? And why not make the argument explicit? That is, we need to know the realization X=x before H* returns a set of parents, and yet that is not clear at all from the notation.

194: “the minimal set”: as mentioned, unicity is not guaranteed.

(6) What is the norm here? 

(7) Why use different types of distributions for (a), (b), and (d)? Why not use the empirical distribution in all three cases?

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
Context-specific (here called "event-level") independence is used to define "event-level causality", which is then used to define a notion of causal feature importance. A two-part neural network architecture is proposed to select the causally important features.

### Strengths
- Context-specific independence can be a valuable tool in explainable AI, and this paper contributes a generally applicable, end-to-end differentiable approach "Hide&Seek" to learn such independences from data.

### Weaknesses
- Early parts of the paper discuss Bayesian network structure learning and causal discovery. However, in the context of the paper, there is a known topological ordering of the variables, in which case the aforementioned problems reduce to feature selection (and indeed, this reduction is made at the start of section 4). I think the message would be clearer if feature selection was discussed from the start. I am also not convinced that this paper manages to "unify" causal modelling and explainable AI, as claimed in the early sections.

### Questions
- In section 4 (and the appendix), it is shown that an earlier approach may suffer from an information leak. But I didn't see a proof in section 5 that the newly introduced method does not suffer from such a leak. Do you believe this to be true?
- "the minimal set of parents" (line 194) may not be unique! For instance if $X_1$ and $X_2$ are independent coin flips and $X_3 = X_1 \wedge X_2$. Which feature should then be called causally important in the different realizations? And does the Hide&Seek algorithm handle this correctly?
- line 212-213: choosing $\lambda$ this way does *not* ensure what is claimed, because a too small $\lambda$ will lead to overfitting. How would you address this?

### Other remarks
- "Bayesian structure learning" has a different meaning than "Bayesian network structure learning", but they are used interchangeably in this paper, and both abbreviated to BSL. I suggest being consistent and always using BNSL.
- line 79: final index i should be k
- line 188-189: "that precede $X_i$ (for the realization $x_{<i}$)" suggests that the predecessor relation may change depending on the realization, but that's not what this is trying to say
- equation (6): the left-hand side should be $pa_i^*(x_{<1}$, because the right-hand side also depends on $x_{<1}$
- equation (6): set size is usually denoted by single, not double vertical bars

### Soundness
2

### Presentation
2

### Contribution
3
