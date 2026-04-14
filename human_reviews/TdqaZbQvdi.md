# On the Relation between Trainability and Dequantization of Variational Quantum Learning Models

- Decision: Accept (Poster)
- Scores: 8, 8, 6, 6

## Abstract
Quantum machine learning (QML) explores the potential advantages of quantum computers for machine learning tasks, with variational QML among the main current approaches.
While quantum computers promise to solve problems that are classically intractable, it has been recently shown that a particular quantum algorithm which outperforms all pre-existing classical algorithms can be matched by a newly developed classical approach (often inspired by the quantum algorithm).
We say such algorithms have been dequantized.
For QML models to be effective, they must be trainable and non-dequantizable.
The relationship between these properties is still not fully understood and recent works raised into question to what extent we could ever have QML models which are both trainable and non-dequantizable.
This challenges the potential of QML altogether.
In this work we answer open questions regarding when trainability and non-dequantization are compatible.
We first formalize the key concepts and put them in the context of prior research.
We introduce the role of "variationalness" of QML models using well-known quantum circuit architectures as leading examples.
Our results provide recipes for variational QML models that are trainable and non-dequantizable.
By ensuring that variational QML models are both trainable and non-dequantizable, we pave the way toward practical relevance.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The authors address a pressing question in the field of quantum machine learning: do there exist "variational" (i.e., trained via gradient descent) quantum neural networks which are both efficient to train and also difficult to classically simulate? The authors answer this question in the affirmative, giving a general method for constructing variational quantum machine learning models with these two properties.

### Strengths
The authors prove concrete, rigorous results, which are hard to come by in the field of quantum machine learning; this is particularly true of concrete quantum-classical separations in machine learning (where the quantum model is assumed to be efficiently trainable). The authors give explicit examples of efficiently trainable quantum networks with provable quantum advantages in learning over their classical counterparts.

### Weaknesses
The authors' Theorem 3 and Corollary 4---which give the recipe for constructing a learning problem solved by a quantum model efficiently trainable via gradient descent, but which is unable to be dequantized---essentially separates these two conditions out, considering a tensor product of two systems where one is trivial and easy to optimize and the other implements a quantumly-easy, classically-hard function, but is untrained. Because of this, it is difficult to know what broader impact this work has on the actual design of variational quantum algorithms (see Questions).

A minor additional weakness is that the authors' proposed criteria for "dequantizable" focuses on the supervised learning setting; the unsupervised learning setting (e.g., sampling problems) may also contain concrete quantum advantages in an efficiently trainable setting, particularly with the current wealth of quantum experiments demonstrating a quantum advantage in sampling tasks (Nature 574, 505; Nature 626, 58).

### Questions
What implications do the authors' results have on the construction of quantum neural network architectures?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper formalizes the relationship between differing ideas in the QML literature regarding different aspects of learning QML models. With these formalizations in hand, the authors then prove that there exist variational QML models which are trainable and non-dequantizable.

### Strengths
- Figures 1 and 3 are good, and help with reader comprehension
- The paper is generally well written and understandable
- The writing style especially is very approachable and the focus on explanation is beneficial to the paper
- The coverage of recent literature is sufficient

### Weaknesses
- This paper seems to want to do two things. After introducing a new representational/formal scheme for QML, it wants to (a) show how other papers fit into this scheme and (b) use this scheme to prove novel results. However, in attempting to do both, results in both being weaker. If the paper were to lean into (a) and be more of a review paper (where the formalize in the synthesis of many previous papers) and give more examples of how the literature fits into this paradigm, it would be good. Or, if the paper were to lean into (b) and emphasize more the research power of this paradigm through novel theoretical contributions, that would also be good. To be concrete for this latter approach, this would largely involve moving 3.3/3.4 to an appendix.
- Figure 2 and Table 1 seem to convey the same information redudantly
- This paper is generally interesting and is a quality paper, but I question whether ICLR is the ideal venue for it and whether there is sufficient novelty of interest to this community
- Not a major point, but the “vari veryational” term doesn’t seem ideal, both for speaking (sorry for the phonocentrism Derrida), but also for writing (with autocorrect/spell check). Maybe a different abbreviation would clarify better?

### Questions
- If “de-” is the prefix for the reverse of something, can non-dequantizable just be “quantizable”?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper tries to establish a relationship between trainability and dequantization in the context of QML models. The authors present a formalization of these concepts and attempt to illustrate conditions under which these properties co-exist.

### Strengths
- The paper commendably formalizes the concept of dequantization, linking several key concepts in QML models, including trainability, dequantization, and classical simulation. This integration provides a valuable framework for understanding QML models.

- The paper is well written and concepts are explained clearly.

### Weaknesses
- From a technical standpoint, the paper's contribution appears limited, primarily synthesizing existing results rather than offering new findings. It attempts to establish connections between different unclear concepts but lacks significant technical contributions.

- The discussion would benefit from a more detailed analysis of existing QML models to determine which categories they fall into. Such a comparison would enhance the paper's relevance and applicability in the field.

- On the practical side, the paper falls short of providing actionable insights or guidelines for designing effective QML models, which limits its utility for practitioners in the field.

### Questions
- It would enhance the paper if the authors could offer insights or preliminary guidelines on designing QML models that are trainable, non-dequantizable, and very variational.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper discusses and clarifies several important concepts in quantum machine learning (QML), including trainability, simulatability, and de-quantization. Building upon the established concepts, this paper introduces a new family of "vari veryational" QML models, a spoonerism of "very variational", that capture the essence of deep-layered QML models and gradient-based training algorithms. It is shown that in this family of QML models, there exist non-dequantizable but trainable models. This resolves an open question that trainability and dequantization are mutually compatible.

### Strengths
- The paper proposes clear definitions for trainability and dequantization using a rigorous learning theory language. This clarifies the vagueness of many seemingly related but not equivalent concepts in quantum learning theory.
- This paper constructed a QML model that is gradient-based trainable but not dequantizable (based on standard cryptographic assumptions).
- This paper provides an extended discussion of several related results in the quantum learning theory (Figure 3).

### Weaknesses
- The QML model constructed in this paper seems a bit contrived. The construction is based on a computationally hard problem, and the proposed training method is quite specific and not able to generalize to other QML designs. I feel this construction is mostly of theoretical interest and has limited connection to practically relevant variational quantum algorithms.
- Several definitions in Section 2 are quite formal and math-heavy, and it’s unclear whether such definitions are truly necessary in light of the paper’s main technical contributions. I feel the theoretical framework may be somewhat excessive relative to the provable results.

### Questions
- What does "gradient-based trainable" mean precisely? Does it mean there is no barren plateau in the sense of Definition 2?
- On page 2, the risk functional is defined over the space $\mathcal{Y}^\mathcal{X}$. Can you explain this notation, specifically, why the data domain $\mathcal{X}$ appears as the power in the label co-domain $\mathcal{Y}$, but not vice versa?
- I am a bit uncertain about the name "vari veryational". Is there further justification for this name, besides it's a spoonerism of "very variational"? It feels a bit uninformative to the general quantum information audience.

### Soundness
3

### Presentation
3

### Contribution
3
