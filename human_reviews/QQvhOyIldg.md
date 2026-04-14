# An $O(k\log n)$ Time Fourier Set Query Algorithm

- Decision: Reject
- Scores: 8, 3, 1, 3

## Abstract
Fourier transformation is an extensively studied problem in many research fields. It has many applications in machine learning, signal processing, compressed sensing, and so on. In many real-world applications, approximated Fourier transformation is sufficient and we only need to do the Fourier transform on a subset of coordinates. 
Given a vector $x \in \mathbb{C}^{n}$, approximation parameters $\epsilon, \delta \in (0, 0.1)$, and a query set $S \subset [n]$ of size $k$, we propose an algorithm to compute an approximate Fourier transform result $x'$ which uses $O(\epsilon^{-1} k \log(n/\delta))$ Fourier measurements and runs in $O(\epsilon^{-1} k \log(n/\delta))$ time. For $\hat{x}$ being the Fourier transformation result, our algorithm can output a vector $x'$ such that $\\| ( x' - \widehat{x} )\_S  \\|\_2^2 \leq \epsilon \\| \widehat{x}\_{\overline{S}} \\|\_2^2 + \delta \\| \widehat{x} \\|\_1^2 $ holds with probability of at least $9/10$, where $\overline{S}$ denotes the complement of the set $S$, i.e., $\overline{S} := [n] \setminus S$.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper studies the design of sublinear time and query algorithms for the Fourier set query problem. In this problem, one is given query access to the entries of $x\in \mathbb{C}^n$, a set $S\subseteq [n]$ of size $k$ and parameters $\epsilon,\delta >0$. The goal is to output a vector $x'$ such that $\|(x'-\hat{x})_S\|_2^2 \leq \epsilon \|(\hat{x})_T\|_2^2  +\delta \|\hat{x}\|_1^2$ where $\hat{x}$ is the Fourier transform of $x$ and $T=[n]\setminus S$. The previous best algorithms for this problem had query complexity $\epsilon^{-1}k$ and runtime $\epsilon^{-1}k \log^{2.1}n \log (R^*)$ ($R^*$ is the infinity norm of $x$) or query complexity $\epsilon^{-1}k\log^2 n$ and runtime $\epsilon^{-1}k \log^{2}n $. The contribution of this paper is to design an algorithm with $\epsilon^{-1}k \log n$ sample complexity and runtime.

### Strengths
The main strengths of the paper are to improve both the sample and time complexity of the original algorithm of Hassanieh et al. using core techniques, such as using HashToBins to isolate frequencies, recover them and iterate, that originate from Hassanieh et al. directly. This thus shows that existing techniques can indeed achieve better results. The algorithm also improves upon the algorithm of Kapralov et. al. in terms of the time complexity by improving log factors and removing dependence on the infinity norm of the vector.

### Weaknesses
One weakness of the algorithm is that it does not explain how the application of the classical techniques in Sublinear Fourier analysis such as HashToBins and iterative frequency recovery are applied differently in this paper compared to existing works to ensure the improvement in the result. Moreover a minor technical limitation of the paper is to obtain worse sample complexity compared to the algorithm of Kapralov, however this is not a big limitation to discredit the novelty of the algorithm in applying existing techniques to refine the algorithmic guarantees.

### Questions
Can the authors please explain briefly what is the difference in the approach of this algorithm, with regards to using HashToBins and iterative frequency recovery after isolating frequencies in each round of hashing and recovering, compared to Hassanieh et al and Kapralov's result ?

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
3

### Summary
Fourier set query is similar in nature to sparse FFTs, except in this manuscript the sparsity set is selected a priori. The algorithm has both sample and time complexity of O(eps^{-1}klog(n/delta)), and guarantees an eps/delta-accurate set of Fourier modes in the query set S with high-probability.

### Strengths
The manuscript looks original and it appears to have improved on both the sample and time complexity of the set query Fourier problem.

### Weaknesses
- The main techniques look similar to those found in Hassanieh et. al. I believe it is mostly technique II that is different from Hassanieh et. al. This comes across as a very carefully performed iteration on the set S, which seems tricky and technical.  From the manuscript, I wasn’t able to follow the reasoning to convince myself that the manuscript is technically correct. 

- The manuscript is poorly written in my opinion. Having informal versions of many of the lemmas is not very helpful. Instead, it would have been better if the manuscript used that space to explain the many technical contribution of the manuscript. 

- I really wished the manuscript had a few sentences about the motivational applications. I am used to sparse Fourier problem, where the query set S is not known in advance, and the algorithm is designed to get both a set S (with |S|=k) and the Fourier modes in S. In that setting, the Hassanieh et. al paper from 2012 takes O(k*log(n)*log(n/k)) in samples and running time. It is also a deterministic algorithm. Query set Fourier, on the other hand, is limited to retrieving specified frequencies without leveraging the full sparsity of the signal.

### Questions
- Can you give me a few motivating applications where the set query Fourier problem appears, not the sparse Fourier problem? 

- How does your algorithm compare to Hassanieh et. al from 2012? 

- Would it be possible to implement your set query Fourier algorithm?  How does it behave under floating-point rounding errors?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
1

### Rating Number
1

### Confidence
5

### Summary
This paper propose an algorithm to compute the approximate Fourier transform on a query set of size $k$. Through the techniques of HASHTOBINS, they obtain the error analysis between  the approximate Fourier transform and Fourier transform,

### Strengths
NA

### Weaknesses
1. This paper is poorly written & presented. A lot of the content can be found in the undergraduate textbook. A substantial part of the results are informal version, say Lemma 6.1 - 6.3. Also, there is hardly any interpretation of the main results. The presentation style does not seem to be serious. 
2. The technical contribution is unclear. Most of the analysis are quite standard. 
3. There is no numerical experiments to verify its application in real-world dataset.

### Questions
See above

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper studies the Fourier set query problem: given a set of indices $S \subset [n]$ and a vector $x \in \mathbb{C}^n$, the goal is to compute the Fourier transform $\widehat{x}$ of $x$ at the set of indices $S$ with precision specified by parameters $\epsilon$ and $\delta$. More formally, the algorithm is required to output a vector $x'$ such that $$|| (x' - \widehat{x})_S ||^2_2 \leq \epsilon ||\widehat{x}_S||^2_2 + \delta || \widehat{x} ||^2_1. $$

Since the complete Fourier transform can be computed in time $O(n \log n)$, it is desired that the running time of the algorithm would be less than this. At the same time, for some applications the sample complexity, i.e. the number of entries in $x$ that are read by the algorithm, is also crucial.

The main contribution of this paper is an algorithm that solves the Fourier set query problem with time and sample complexity $O(\frac{k}{\epsilon} \log n)$, achieving an improvement over previous work.

### Strengths
The authors prove all of the made claims, including the main result. Detailed proofs for all of the original theorems and lemmas can be found in the appendix.

### Weaknesses
The greatest weakness of the paper is the lack of clarity, which greatly impedes assesment of its other merits. Some of the concerns are:

1. This paper uses a lot of prior work, and it is not made clear what the improvements in techniques are, or sometimes what even are the techniques. For example, in section 4 the paragraphs discussing techniques 3 and 4 consist mainly of an outline of the proof. Concrete techniques are not highlighted, and it is hard to understand which steps are standard  in the literature, and which ones are the contribution of the paper.
As far as I am aware, the only thing stated as original contribution is the algorithm itself and the technique 2.

I think that the paper should be rewritten in a way such that the key critical insights that led to the improvement of the algorithm become identifiable. My suggestions are 
 - Extract the techniques 2 and 3 from the proof structure to show what do they allow you to do, preferably by formatting them as a lemma or a theorem. Otherwise, do not call those paragraphs "techniques", just a "proof overview".
 - For techniques 2 and 3, explicitly mention if the proof steps are an entirely novel contribution, and if not list works where they were borrowed from and discuss how your use of them is different from prior work.
2. Explanation of how the algorithm or proofs work is lacking intuition. The majority of the paper, starting from section 3, is made up of dry and formal derivation of the main result. 
I would suggest:
 - Adding informal but detailed explanations of what each section does at its start, starting with section 3.
 - Add brief "intuition paragraphs" after each major theorem or lemma to explain its significance in plain language

As a consequence, in my opinion the significance of this paper in its current state is rather low. While the claimed result appears to be interesting, I doubt that the paper itself right now would of use for anyone, as it is very difficult to understand.

### Questions
I have previously reviewed this paper for a different conference, and as far as I can tell the only difference between the two versions is in Section 4, where some explanatory text have been added to techniques. In my opinion, it is a step in the right direction, but it does not resolve the concerns that I have raised back then about the readability of the paper.
In the rebuttal for the previous conference, the authors also mentioned in their reply specific steps that they would take to improve the readability, which have been partially implemented in Section 4. I think that doing the rest of them will be greatly beneficial for the state of the paper.

### Soundness
2

### Presentation
1

### Contribution
2
