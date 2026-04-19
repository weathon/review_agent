# Declarative characterizations of direct preference alignment algorithms

- Decision: Reject
- Scores: 5, 6, 8, 5

## Abstract
Recent direct preference alignment algorithms (DPA), such as DPO, have shown
great promise in aligning large language models to human preferences. While this has motivated the development of many new variants of the original DPO loss, understanding the differences between these recent proposals, as well as developing new DPA loss functions, remains difficult given the lack of a technical and conceptual framework for reasoning about the underlying semantics of these algorithms. In this paper, we attempt to remedy this by formalizing DPA losses in terms of discrete reasoning problems. Specifically, we ask: Given an existing DPA loss, can we systematically derive a symbolic expression that characterizes its semantics? How do the semantics of two losses relate to each other? We propose a novel formalism for characterizing preference losses for single model and reference model based approaches, and identify symbolic forms for a number of commonly used DPA variants. Further, we show how this formal view of preference learning sheds new light on both the size and structure of the DPA loss landscape, making it possible to not only rigorously characterize the relationships between recent loss proposals but also to systematically explore the landscape and derive new loss functions from first principles. We hope our framework and findings will help provide useful guidance to those working on human AI alignment.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The submission devises a new framework that can translate DPA loss functions into a logical characterization. As a byproduct of that framework, the authors establish a double-exponential upper-bound on the number of different DPA loss functions that can be represented in their model. The submission is fairly well-written, but it is clearly targeted at experts in the specific subfield as, e.g., the introduction assumes knowledge of DPA and DPO and is not very accessible to the general ICML audience. 

I have two main concerns:

1) The contribution is comparatively weaker than what one would expect from a typical ICML paper. Of course, contributions can come in different forms, but aside from new ideas fully-fledged ICML papers typically support these ideas with a technical contribution, such as experimental evaluations or non-trivial mathematical proofs. The submission, however, lacks the former as well as the latter (all statements are established as direct observations or via essentially trivial proofs which do not seem to require novel insights or ideas). 

2) While the submission identifies a large number of mathematically well-defined DPA loss functions, it is less clear what is the envisioned contribution to the ICML community. After all - unlike when one establishes, e.g., theoretical upper/lower bounds or settles the computational complexity of fundamental problems - here there was no doubt that many different DPA loss functions exist. The authors claim that their results can be used to "map out" and find loss functions with better properties, but that is left entirely for future work (see Section 6.2).

Given these concerns, I feel that the submission would benefit from diving deeper into the topic and presenting a more well-rounded and thorough contribution. Note that space constraints are not a major factor here yet: the current submission spends a lot of space discussing tangential remarks which could easily be partly or wholly moved to the Appendix (see, e.g., the end of Subsection 6.1).

### Strengths
The submission is well-written and the research direction seems sound.

### Weaknesses
See the Summary.

### Questions
-How important is it for the formalization to follow the assumption about DPA losses suggested by Tang et al. (arxiv 2024)? As far as I am aware, that article was not yet peer-reviewed and it is hence not clear to me how well-established this particular formalization of DPA losses is (and the submission does not attempt to justify this on its own).

### Soundness
3

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
4

### Summary
This paper addresses the challenges in understanding and developing direct preference alignment (DPA) loss functions, commonly used to align large language models with human preferences. Current DPA methods, like DPO, show promise but lack a conceptual framework for analyzing and differentiating variants. To address this, the authors propose a formalism to characterize DPA losses as discrete reasoning problems, enabling a systematic derivation of symbolic expressions that define their semantics. This approach reveals the extensive structure within the DPA loss landscape, showing a doubly exponential number of definable variations based on unique predictions. The framework highlights formal relationships, such as logical entailment and monotonicity, providing insights into efficiently exploring new DPA losses by modifying and testing established loss functions. This formal view aims to guide further development in human-AI alignment research.

### Strengths
The paper’s strengths include a novel technique that clarifies the semantics of DPA losses through logical formulas over Boolean propositions, addressing a key gap in understanding what these losses capture and how they interrelate. The innovative decompilation procedure enables deriving symbolic formulas for complex loss functions, offering a structured view into the DPA loss landscape. This approach empowers practitioners to systematically explore new loss functions, advancing both theoretical insights and practical tools for human-AI alignment.

### Weaknesses
The paper could be strengthened with more real-world examples demonstrating the practical relevance of formalizing DPA losses as discrete reasoning problems. While the formalization offers a structured approach, it’s primarily theoretical, and its effectiveness remains unproven. The claim that new losses derived from this framework are superior is speculative, as no substantial evidence is provided to show that these new losses outperform existing ones. Further empirical validation is needed to confirm the benefits and applicability of these new loss functions in real-world settings. Additionally, the hypothesis around the "constrainedness" of a loss function as a predictor of its success is only preliminary, requiring more in-depth experimentation.

### Questions
1. Can you provide additional real-world examples of applying formalized DPA losses as discrete reasoning problems to clarify the framework’s practical relevance?
2. While you suggest this framework aids in finding improved loss functions, is there empirical evidence or additional experiments comparing these new losses to existing DPA losses?
3. How does the complexity of your decompilation procedure scale with larger models or more complex DPA losses? This would clarify its feasibility for large-scale applications.

### Soundness
3

### Presentation
3

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
The paper introduces a novel framework for analyzing the space of DPA-like losses by systematically deriving a symbolic, logical expression characterizing its semantics. Different DPA losses can be derived thanks to such mapping, thus providing a comprehensive overview of the landscape. The authors do an excellent job of formalizing their framework, which gives researchers a new, fresh perspective on how to analyze the plethora of what they call successful DPA losses in the literature. Intuitively similar losses found in the literature can now have a place where analysis is done through formal methods if the paper is accepted. Be aware that my evaluation could have been overly optimistic, given my expertise; nonetheless, the paper should be accepted for me based on the novel contribution and rigorous mathematical treatment. As such, I would give a 7 (disabled by the system) instead of an 8, while a 6 seemed too pessimistic.

### Strengths
- **S1:** Precise related work.
- **S2:** Strong mathematical formulation, which I greatly enjoyed.
- **S3:** New perspective on neural-symbolic interplay.
- **S4:** Great presentation and structure of the work.

### Weaknesses
- **W1:** Given my unfamiliarity with the literature on DPA and similar, there seem to be exponential blowups. For example, to compute Eq 2, the weighted model counting must enumerate all $2^n$ propositional models ($\mathbf{w}$).
- **W2:** I would have given more context around the notation $y_w \succ y_l$ for those unfamiliar with the literature like me. I have interpreted $\succ$ as "the winner is preferred to the looser", but again, this may be the wrong interpretation. Either way, please clarify.
- **W3:** Some references are wrong. For instance, the reference to Table 7 does not exist in line 416; maybe it should be Table 5 (from the Appendix). Similarly, the caption of Table 4 references Algorithm 5.1, but 5.1 is a Section.

Minors:
- line 176: an a variant --> and a variant
- line 385 (and similar): Before the proposition's statement, the notation does not use parentheses around the superscripts, while the statement uses them (also in the Appendix). Please fix it for better readability.
- line 390: prefrence --> preference
- line 476: CEUNL --> CEUnl
- line 482: is much to learned by --> is much to *be* learned by?
- line 522: exactly these the losses --> exactly the losses?
- line 728: Table 5 is referred to in the proof, but Table 5 is the one with the translation rules
   - As a meta-observation, double-check all the other references (`\ref{}`). Some are okay, but some are not (and I tried to point to some of those).

### Questions
- **Q1:** I didn't get the number 4 in the double exponential form, i.e. $4^{2^n}$. Could you please be more specific or provide more context?
- **Q2:** Algorithm 1 uses "Simplify" to minimize (propositional formulas). As far as I am aware, the problem of minimizing propositional (logic) formulas by preserving equivalence is NP-hard. Thus, am I missing something here? Please clarify.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The authors investigate losses for preference alignment. They analyze existing DPO functions with an aim to symbolically extract their semantics, and also investigate compiling DPO losses from symbolically given knowledge on preference structures.

### Strengths
- The authors systematically investigate an interesting and relevant problem of semantically understanding and constructing preference alignment loss functions.
- The paper proposes a simple  algorithm to compile DPA loss to a logical expression
- They introduce a logic for modeling preferences, that allows to create new loss functions for a given preference structure.

### Weaknesses
- This could be due to my relative lack of expertise in the field of the paper. But the lack of any running example, and experiments, makes the presentation quite divorced from the original motivation of preference alignment in AI models.
- Hence, the larger potential utility of the framework is not clear to me.

### Questions
- Could you please provide a toy example, and an analysis of this example for each of the introduced contribution of the paper?
- What could be an empirical setting, where your proposed framework could be investigated?
- Could you please elaborate on "While this can be remedied by modifying the SL to involve counting multiple formulas as in Rescher (1967), we instead define a relational structure called a preference structure that allows us to capture the semantics of losses in a modular fashion using a single propositional formula coupled with auxiliary constraints. Such a structure, which is based on a novel construction in propositional logic, will later make it easy to cleanly characterize different DPA losses and devise new variants through manipulation to their constraints." --- It is not clear to me, why the new method you propose is motivated by this. Aren't you compiling your loss to an SL as well? what are the main differences?

### Soundness
3

### Presentation
2

### Contribution
2
