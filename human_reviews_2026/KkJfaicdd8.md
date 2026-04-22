# Symmetric Beauty of Hopfield Neural Networks with an In-Depth Analysis and Insights

- Avg Score: 3.00
- Decision: Reject
- Scores: 2, 2, 2, 6

## Abstract
The Hopfield Neural Network (HNN) comprises \\(N\\) binary neurons, yielding a state space of size \\(2^N\\). Traditional update rules leave a neuron's next state unspecified when its input summation is zero, leading to symmetry-breaking artifacts and spurious cycles. To remedy this, we introduce the **Rectified Update Rule**, which retains each neuron's prior state in such tie scenarios, thereby restoring symmetry and ensuring stable convergence. Building upon this, we reformulate the **Hamming-Distance-Aware Rectified (HDAR) Update Rule**, considering the Hamming distance when memorizing two messages. This rule preserves full symmetry among the two memories and their negations and yields a complete taxonomy of dynamic regimes: convergence, self-cycle, hetero-cycle, and symmetric cycle. Importantly, we encapsulate these dynamics in **two central theorems**—one characterizing single-message behavior and another for dual-messages regimes—with full proofs in Appendix. From these theorems, we derive corollaries that precisely quantify the counts and conditions of convergent versus cyclic states as functions of the network size and the Hamming distance. Extensive simulations, spanning exhaustive enumeration and Monte Carlo sampling, confirm all theoretical calculations.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors introduce a modification to the Hopfield model with a sign function nonlineartiy that takes into account ties, and then they full characterize the behavior of the model when there are either 1 or 2 memories.

### Strengths
The problem they considered was very clear and explained well. As far as I could tell all their results, including the proofs, were correct, and they were backed up by simulations.

### Weaknesses
I personally could not understand why the problem they considered was especially relevant. The motivation in the introduction is that deep networks need to be interpretable, and the "Transformer’s attention mechanism is formally equivalent to the continuous-state
update rule of modern Hopfield networks", therefore it makes sense to study Hopfield networks. However, the network they studied was not the one that was equivalent to Transformer’s attention mechanism, and they looked only at 1 and 2-memory networks.

### Questions
The main thing that would change my opinion would be an explanation of why this problem is relevant.

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
An analysis of the Hopfield model using synchronous updates, showing the spectrum of dynamic behaviour for a single or two patterns stored in the network. Also suggest a modified dynamic rule, based on the Hamming distance.

### Strengths
* A thorough analysis of the dynamics for the original and modified ("Hamming-based") dynamics.
* A beautiful visualization of the spectrum of dynamic conditions (Figure 1 and 2).

### Weaknesses
* Per the authors' summary, the problem of a Hopfield model with synchronous updates hasn't been under active research between 1987 and 2022. Possibly because those systems cannot act as a useful memory device. The authors should convince the reader why they seek to reignite this line of research: why is it interesting to do the suggested analysis?
* Table 1 (left) should have been a sentence in the figure caption. Table 1 (right), 2, and 3 should have been plots.
* The exposition taking the first page in its entirely does not contribute to the paper and is not accepted in modern scientific writing.
* The contribution of the "Rectified Update Rule" is so small you don't need to specify it.

### Questions
* Can the suggested network act as a memory device? How many memories can it store for 1K neurons?

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper “Symmetric Beauty of Hopfield Neural Networks: An In-Depth Analysis and Insights” investigates the relationship between the dynamic behavior of Hopfield networks and the properties of the stored memories, particularly their mutual distances in configuration space. The authors present an analysis that reveals a rich convergence phenomenology under a novel dynamical rule, which accounts for the possibility that a given initialization may be equidistant from multiple memories.

### Strengths
The main strength of the paper is the exact and controlled analytical framework where Authors work.

### Weaknesses
I believe that, in its current form, the paper has several weaknesses, which I outline below:

1. The research question—deriving the dynamic phenomenology of simple Hopfield networks—is clearly stated. However, the paper fails to clearly articulate the broader significance of this topic for the fields of associative memory and machine learning.

2. Although the Hopfield model has been extensively studied, it continues to raise interesting and relevant open questions, precisely because of its elegant and simple formulation. The problem addressed by the authors is indeed of interest, but the paper should devote more space to convincing the reader of the importance and novelty of their contribution. The main threat to the relevance of the results is the fact that, for a large size N of the system, sampling initialization that are exactly equidistant to several data-points - i.e. that yield a zero local field - is exponentially rare in N. For this reason it is not clear how the phenomenology explained by the paper should count in actual implementations of the model and why it should be so theoretically insightful.  

3. The overall presentation of both the content and the results—particularly the experimental results—is very weak and requires substantial revision. In addition, the paper lacks mathematical rigor, and the current writing style does not meet the standards of a scientific publication. For instance, the conversational exchange between a student and a large language model in Section 1 is not appropriate for this context. Furthermore, the title of the article should be updated to a more professional version and perhaps one that is more specific to the topic covered. 

As a result of these issues, I believe the paper is not yet ready for publication.

### Questions
I will now proceed to ask questions and raise issues that I would like the authors to address in order to both finalize my personal judgment over the paper and improve the manuscript. I will divide the issues into Major and Minor ones for an improved readability of the revision. 

**Major issues:**

1. [120-121] how can $X$ reach zero value in large $N$ limit? How is this relevant to the dynamic behaviour of the network? As already mentioned in the "Weaknesses" Section, Authors should argue around the validity and relevance of their results by addressing this aspect (you can answer only once to both questions). 
2. How does the dynamic phenomenology generalize to a larger amount of memories or to the case where the memory load scales as $p = \alpha N$, as in the usual Hopfield setup? Can the current approach explain the rich dynamic phenomenology of stable fixed points and limit cycles convergence observed in this regime? See works already referenced in Peretto (1984) and Section C3 of the paper. 

**Minor issues:**

1. [040-041] References to the student’s episode, and definitely rephrase that part to include it in a scientific journal. 
2. [057-058] Wrong reference to Hebb (2005?) across all the paper. 
3. [059-060] Authors should be more specific on the type of dynamics to which this line refers to. Asynchronous dynamics over hopfield networks - that are symmetric in the couplings by construction - can only end up into stable fixed points. On the other hand, asynchronous dynamics can also lead to limit cycles of length 2 (Peretto (1984), Folli et al. (2018), Hwang et al. (2019)). I ask the authors to provide for more detailed reference about the fact that the number of stable fixed points is smaller than the 2-limit cycles as stated.   
4. [097-098] It would be convenient to replace capital $W$ with small $w$, since couplings are later referred to as $w_{ij}$. Or at least specify that $w_{ij} are the elements of $W$. 
5. [100-101] Authors may replace “quantity of” with “number of”.   
6. [129] Authors may remove “elegant”. 
7. [130] $U$ should be defined with respect to v used in Eq. (2). Also it appears as sometimes patterns, i.e. network configurations, are referred to as $U$, some other times as $V$. All the notation in the paper should be uniformed. 
8. [131] Authors may split the fixed point and cycle definition into two definitions, and also rephrase the definition. For instance “and t is the minimal value” is not a clear expression. 
9. [135-136] Authors should rephrase Theorem 1 as well for a better clarity.   
10. [144-145] and elsewhere in the paper: replace $\overline{\xi}(\overline{\xi}=-\xi)$ with $\overline{\xi}=-\xi$ . 
11. All quantities $C_{ab}$ in Corollary 2 and in the Appendices are not previously defined. May the authors explain why and how they are computed ?
12. Should not Corollary 3 be a condition on $r_{12}$ instead of $r_{ccd}$ ? How do $r_{ccd}$ and $r_{max}$ scale with $N$? In fact no information is provided about the value of $r_{ccd}$ except for condition (62). You can either provide for an estimate of $r_{ccd}$ as a function of $r_{12}$, and then Corollary 3 is well defined, otherwise you provide for a bound in $N$, and at that point Corollary 3 should turn into a condition on the distance between the memories, that is also the way it is naturally written.  
13. Can you draw $r_{ccd}$ and $r_{max}$ in Figure 2? For a more clear explanation. Authors may also include a series of images where the mutual distances between memories change, and show how the dynamic phenomenology is changing. 
14. [414-415] and [421]: I guess there is a typo and that authors have performed MCMC sampling when $N ≥10$. 
15. The content of table 1 in Section 5.1 is not clear from both the table and the main text. For large networks, it should expressed more clearly that authors generate patterns at $r/N =\frac{1}{2} + O(N^{-1/2})$ and then change $N$ and that convergence ratio is then estimated. For small networks the procedure is not clear: should not the distance $r$ also be fixed, so to vary $N$ ?
16. The same comment above holds for Section 5.2. Also, the reader cannot know in what region of Figure 2 the patterns generated in Table 2 and 3 are. As a consequence the reader does not know what to expect. This should be clarified, as in general what the tables actually show.  
17. Bibliography more homogeneous (i.e. all first names extended or shortened). 
18. Errors in experiments in Appendix B.3 have an enormous quantity of digits that should be correctly approximated.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work introduces the Rectified Update Rule for Hopfield Networks, which resolves tie scenarios by retaining a neuron's prior state. This restores network symmetry and ensures stable convergence. When memorizing two messages, incorporating Hamming distance into the neuron's transition dynamics preserves full symmetry between the two memories and their negations. This approach yields a complete taxonomy of dynamic regimes: convergence, self-cycles, hetero-cycles, and symmetric cycles.

### Strengths
1. This paper proposes the Rectified Update Rule,which retains the neuron’s previous state in such tie cases and offers insights potentially applicable to activation function design in DNNs.

2. For two-message memorization, this paper intorcuces a Hamming-Distance-Aware Rectified (HDAR) update rule. This rule induces perfectly symmetric network dynamics, categorizing them into four regimes: convergence, self-cycle, hetero-cycle, and symmetric-cycle.

### Weaknesses
1. Since a Transformer's attention is equivalent to a modern Hopfield network, we can use Hopfield theory to understand Transformer models. This movtivation is very intersting. However, it is unclear that what new understanding for the Transformer models using Hamming-Distance-Aware Rectified (HDAR) update rule.

2. The authors suggest that these insights could inform the design of novel activation functions for Deep Neural Networks (DNNs). Could you provide potential directions or concrete examples of how this might be achieved?

3. The presented results are based on a synchronous update rule. How would these results differ under an asynchronous update scheme?

4. The experimental results are challenging to interpret. Could the authors provide low-dimensional (e.g., 2D or 3D) visualizations to clarify the findings?

### Questions
please see weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
