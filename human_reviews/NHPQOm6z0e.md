# On the Convergence of Tsetlin Machines for the AND and the OR Operators

- Avg Score: 5.50
- Decision: Reject
- Scores: 3, 3, 8, 8

## Abstract
The Tsetlin Machine (TM) is an innovative machine learning algorithm rooted in propositional logic, achieving state-of-the-art performance in various pattern recognition tasks. While previous studies analyzed its convergence properties for the 1-bit and XOR operators, this work extends the analysis to the AND and OR operators, completing the study of fundamental digital operations. Our findings demonstrate that the TM almost surely converges to reproduce the AND and OR operators when trained on noise-free data over an infinite time horizon. Notably, the analysis of the OR operator uncovers a distinct property: the ability of the TM to represent two sub-patterns jointly within a single clause, contrasting with its behavior in the XOR case. Furthermore, we investigate the TM’s behavior for AND/OR/XOR operators with noisy training samples, including mislabeled samples and irrelevant inputs.  With wrong labels, the TM  does not converge to the intended operators but can still learn efficiently. With irrelevant variables, the TM converges to the intended operators almost surely. Together, these analyses provide a comprehensive theoretical foundation for the TM's convergence properties across basic Boolean operators.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper examines Tsetlin Machines (TMs), a machine learning architecture characterized by notable attributes such as transparent inference and a hardware-friendly design. While recent literature has established convergence properties for the identity, negation, and exclusive-or operators, this study completes the analysis by establishing the convergence of the “AND” and “OR” operators.

### Strengths
The authors demonstrate that TMs can almost surely learn the AND and OR relations with the simplest structures, under noise-free conditions.

### Weaknesses
**Audience:**
The paper primarily focuses on Tsetlin Machines, particularly their capability to learn logical concepts such as AND and OR in a noise-free environment. However, it does not address other potential approaches that can also learn these simple concepts under similar conditions. Consequently, the paper seems tailored to a niche group of researchers invested in Tsetlin Machines, raising doubts about its relevance to the broader ICLR community.

**Significance:**
The introduction highlights that the main motivation for studying Tsetlin Machines is their interpretability, as their inference is explainable and their learning process is transparent. A crucial question arises: "Are there other interpretable models and transparent algorithms capable of learning logical concepts under noise-free conditions?" The clear answer is yes. Noise-free concept learning is merely equivalent  to "proper and realizable PAC learning," which has been thoroughly explored since the 1980s. Notably, when the target function is a conjunction, disjunction, or exclusive disjunction of literals over two features, the class of 2-DNF formulas covers all these target concepts. Additionally, 2-DNF is both properly and efficiently PAC learnable in the realizable setting, with a sample complexity that is logarithmic in the input dimension - see e.g. landmark papers such as Valiant (1984) and Haussler (1988), and Shalev-Shwartz and Ben-David’s book (2014). Finally, set cover algorithms for learning disjunctions, and more generally k-DNF, are entirely transparent (see e.g. Marchand and Shawe-Taylor, JMRL 2002).

Even if we focus solely on the results concerning Tsetlin Machines, it seems that Jiao et al. (2022) have already established a foundation for learning binary relations, particularly the XOR operation. As a result, the current study, which investigates the convergence of other binary relations (AND, OR), appears to have limited significance, despite some technical differences. Additionally, since the learning process assumes ideal, noise-free conditions without irrelevant variables, this contribution is relatively minor.

**Clarity:**
The paper presents significant readability challenges due to issues with formatting and punctuation. Specifically, citations should be enclosed in brackets, and punctuation should be used sparingly. For instance, the sentence “In TM, after training, a certain clause, in propositional logic, is a conjunction of literals [...]” (Line 32) notably impacts the paper's clarity.

Additionally, for the ICLR community, it is crucial to provide a clear background on these architectures. Unfortunately, Section 2 falls short in this respect. Key concepts such as Tsetlin Automata (TA) and the notion of a team are mentioned without adequate explanation. Actually, I had to consult the papers by Abeyrathna et al. (2021) and Jiao et al. (2022) to fully understand these concepts.

### Questions
In light of the above comments: 
* Can the convergence proofs for AND and OR be extended to noisy/agnostic settings? 
* Alternatively, do these convergence proofs still hold in the presence of many irrelevant variables?

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
2

### Summary
The paper aims to show the convergence of Tsetlin machines (TM), a relatively new machine learning approach based on Tsetlin automata (TA), for AND and OR operators. After previous work has done work on the convergence for the XOR operator, as well als the IDENTITY and NOT operators, this fills the gap to show convergence for standard binary operators.

### Strengths
The problem discussed in the paper is a natural continuation of previous works and, based on this line of research, seems to be of interest for the community. The relevance for the paper is hinted at by citing several use cases for TM.

The structure of the paper is straight forward, with each of the two main sections being dedicated to one of the two operators for which convergence is discussed.

I like the loop-back to the previous work on the XOR-operator, establishing a more detailed comparison to other recent results.

### Weaknesses
From this paper, it is very hard to grasp the concept of how TM are used for machine learning applications for anyone not familiar with TM. The usefulness of the main theoretical results of the paper -- the convergence of TM for the AND and OR operator -- also very much remains unclear since it is not discussed.

Also on a technical level this paper does not convey how TM and TA work in the first place. Despite reading through section 2 and appendix A, I did not gain an understanding of what a TM is. To exemplify this, even the start of the first sentence in the preliminaries ("A TM that is to learn the sub-patters of class $i$ is formed by $m$ teams of TAs in the form of clauses, $C^i_j(X), j=1,2,...,m$, where $X$ is the input of a TM, and $X=[x_1,x_2,...,x_o],x_k\in ${$0,1$}$, k=1,2,...,o$.") raises several questions that are never addressed in the paper:

* What is a class here?
* What do you mean by sub-patterns of a class?
* How is a TA even defined?
* From what I read from related work, a TA is a probabilistic automaton. How is it represented by a clause?
* What is a team of TAs? Is it different to just a set of TAs (and if yes, in which way)?
* What are the $x_k$ in the input to a TM?

Many of the definitions, e.g. the definition of $\xi^i_j$ (line 90) or the concept of polarity (line 102ff), are not given in a formal way but only in plain language, making the theoretical foundation of the paper very imprecise.

Most importantly, "convergence" is neither defined nor explained for TM. This makes one of the main statements of the paper, Theorem 1, impossible to understand from the paper alone.

For future revisions I **highly** suggest to make the paper self-contained and make sure that anyone not familiar with TM (which I imagine will be a large fraction of audiences) is able to follow the ideas of the paper. I also want to mention that large parts of the preliminaries including figures, are copied from related work, such as Jiao et al. (2022), Zhang et al. (2022) and Granmo (2018). While they are properly cited and this only affects preliminaries (and thus I do not see concerns regarding plagiarism), I think it would be highly beneficial for this paper to set up its definitions, tailored to the setting of this paper, in order to enhance understandability.

Based on the lack of formal definitions I was not able to follow sections 3 and 4 and cannot give detailed feedback on it.

At the end of section 4, experiments validating the theoretical results are mentioned. However, discussion on the experiments are fully shifted into the appendix. In my opinion, they should also be discussed in the main part of the paper.

### Questions
Can you clarify the questions regarding the theoretical background I raised in the bullet points in the weaknesses section? Should these questions have been resolved from any part of your paper (if yes, where?), or do you assume these as a background a reader should have?

In Eq (1), what is the semantic difference between the testing and training formula? The usual convention is for the empty conjunction to be defined as true, if you differ from that this should be mentioned.

Can you give an example for a machine learning task where TM would be applicable? What would be the TM input and output in that example and how would the TA within the TM look like?

How do the theoretical results you show in this paper affect the use of TM in practice?

minor comments:

* the citation style without parentheses is strenuous to read, please use \citep (see point 4 in ICLR style guidelines)
* line 56: "...while this paper searches for absorbing states" -- what does "this paper" refer to in this sentence? From the grammar I would assume your own submission, but form context rather the paper by Zhang et al. (2022) or Jiao et al. (2022)
* line 64 "learnt" -> learned

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper's main contribution addresses the missing piece for TM's convergence: AND and OR operators. Previous studies have analyzed (and published after peer review) the 1-bit NOT and IDENTITY operators and the 2-bit XOR case. This paper fills the remaining gap (as far as I understand) in 2-bit AND and OR operators. The final result of this paper is to complete the bigger picture of logical operators interpreted over the Boolean algebra. The paper is already in a solid position to be accepted, but minor changes could improve the overall comprehensibility.

### Strengths
- **S1:** Clear mathematical formulation.
- **S2:** Precise collocation in the relevant literature, and therefore, the paper addresses a novel problem
- **S3:** Great illustrations to help the reader.

### Weaknesses
- **W1:** As far as I understand, the results are limited to 2-bit cases. This limitation is essential as real-world data rarely have only 2-bit input.
- **W2:** I may be wrong, but perhaps the interdependence between clauses is the reason previous proofs and those presented in this paper have not generalized to >2-bit cases. 
- **W3:** To learn TMs, non-binary datasets must necessarily be transformed into binary ones. As far as I understand the literature, further research is needed to address this issue. Applying naive techniques to binarize datasets could potentially introduce biases during the transformation.
- **W4:** After a quick search on the Internet, I found an arXiv version of this paper with the same name and typos (see https://arxiv.org/abs/2109.09488). For example, search for ".)." (i.e., extra ".") in both PDFs and there will be a perfect match for the same typos (in the four cases). As such, the authors are likely the same. I am not sure if that was the original intention, but nevertheless, this strategy does not entirely fulfill the double-blind process.

### Questions
- **Q1:** What are the challenges of generalizing this result and the previous ones for XOR, AND, and OR to the >2-bit cases? 
- **Q2:** What are the challenges when having numerical data as inputs? How do you suggest handling non-binary datasets in TMs? Specifically, what techniques could binarize data while minimizing potential biases?
- **Q3:** Have you considered generalizing beyond propositional logic? I am fond of logic, and it would be nice to know if the authors have in their research agenda to extend their results to more expressive logics, such as modal and/or first-order logic. This aspect is also essential as a considerable amount of real-world data comes from signals hardly modeled using static Boolean vectors/masks.
- **Q4:** I don't understand why the authors claimed in Theorem 2 that "[...] the OR logic [...]" when their result applies to 2-bit inputs only. Indeed, Equation 4 (right before Theorem 2) presents four 2-bit inputs. Am I missing something here? Consider clarifying in Theorem 2 that the result applies specifically to 2-bit OR logic, or explain how it generalizes to n-bit cases if that's the case.
- **Q5 (minor):** Are there any reasons why you have used a poorly formatted citation style throughout the paper? I would have expected to see, e.g., (Granmo, 2018) instead of Granmo (2018).

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper studies Tsetlin Machines (TMs). It builds on earlier papers, and in particular on an earlier paper that shows that clauses will converge almost certainly to the XOR logic operator. In this paper, the author(s) intend to answer the same question for the AND and OR operators.
The author(s) notes that upon completing this project, the convergence of all basic operators in Boolean algebra will be established. This certainly makes the questions relevant, and the paper does indeed address these questions.
The author(s) gives proofs for their claims, both in detail in the appendix, as well as the idea in the main text (wherever appropriate). Because of this, and because as far as I can tell the proofs seem correct, their claims are well supported.
The new knowledge that this paper reports is certainly interesting for anyone who studies or uses TMs. Therefore, this paper is relevant for the ICLR community.

### Strengths
This paper has a very clear project, which I believe is important, and relevant to the ICLR community. The paper answers their two questions, namely: "do TMs converge on the AND and OR operator?" positively, by giving proofs of these claims. The proofs seem correct, but I was unable to check all the details.
I appreciated the choice of the author(s) to include "proof ideas" in Sections 3 and 4.
The writing and presentation is mostly very good. I have some slight suggestions for improvement, which I will list in the section "Weaknesses".

### Weaknesses
I think this is a good paper, so my weaknesses are less important than the strengths I identified in the section 'Strengths".

The main weakness, in my opinion, is that the paper does not convey the idea behind a TM well. What are the main ideas behind a TM? What are they used for? Being a non-expert, I had to look this up.
I understand and appreciate that it might not be possible to accommodate for this in general due to page limitations, but I do recommend that the author(s) streamlines some of the presentation, in particular:
- On line 052, the author(s) talks about the hyperparameter T. A bit later, on line 105 they talk about a threshold Th, which later turns out to be related to T. Later on, they mostly talk about T again, and relate this twice to Th. It is still not clear to me how to think about Th and T exactly. For instance in Eqs (2) and (3) it would be useful to know what T is. I recommend to make this more clear.
- On Lines 139 – 147 the authors talk about the two types of feedback they generate. It would be helpful to know an (informal) argument about why these feedbacks precisely are used.

I end this section with a list or minor remarks and typos:
- Line 172: It would be useful to indicate the start and end of the proof idea. As I said earlier, I do like the choice of including the proof idea in the main text.
- Line 226: "we can concluded that" shoud be "we can conclude that".
- Line 228: "we must frozen TA3" shoud be "we must freeze TA3" or something along those lines. 
- Line 290: "samples following Eq. (8) is given' should be "samples following Eq. (8) are given'.
- Line 291: "the absorbing states for Eq. (t) disappears" should be "the absorbing states for Eq. (t) disappear".
- Line 452: I believe that "We conjuncture" should read "We conjecture".

### Questions
A question I wonder about is this. It seems to me that the analysis for the OR statement is much harder than that for the AND statement.
Since the convergence for the XOR statement is already established earlier, and for the AND statement in Section 3 of the current paper, could this be used to derive the convergence for the OR statement using a OR b iff (a XOR b) XOR (a AND b), in a much quicker fashion? Perhaps there are technicalities that prevent such an inference, but it would be interesting to know which they are, and if they could be circumvented.
The currect Section 4 gives us more insight into the hyperparameter T, so I do not suggest to replace this section. I am merely interested in whether such inferences are possible.

### Soundness
4

### Presentation
3

### Contribution
4
