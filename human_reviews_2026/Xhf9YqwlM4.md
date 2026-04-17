# Tokenisation over Bounded Alphabets is Hard

- Decision: Accept (Poster)
- Scores: 8, 2, 8, 8, 6

## Abstract
Recent works have shown that tokenisation is $\mathsf{NP}$-complete. However, these works assume tokenisation is applied to inputs with unboundedly large alphabets—an unrealistic assumption, given that in practice tokenisers operate over fixed-size alphabets, such as bytes or Unicode-characters. We close this gap by analysing tokenisation over bounded alphabets, considering two natural variants: bottom-up tokenisation and direct tokenisation, where we must, respectively, select a sequence of merge operations or a vocabulary whose application optimally compresses a dataset. We prove that even with binary alphabets, both variants are not only $\mathsf{NP}$-complete, but also $\mathsf{APX}$-hard and thus admit no polynomial-time approximation scheme (unless $\mathsf{P}=\mathsf{NP}$). We further show that direct tokenisation remains $\mathsf{NP}$-complete even when applied to unary alphabets. These results establish that the computational intractability of tokenisation is not an artifact of large alphabets or complex constructions, but a fundamental barrier. Overall, our results explain why current practical algorithms such as BPE and UnigramLM are heuristic, and point toward approximation algorithms being an important path going forward for tokenisation research.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper explores the computational complexity of tokenisation when restricted to bounded alphabets, addressing a critical gap in prior research that focused on unbounded alphabets.
The authors analyze both direct and bottom-up tokenisation and demonstrate that these problems remain NP-complete even for binary and unary alphabets. Furthermore, they prove that neither admits a polynomial-time approximation scheme (PTAS) unless P = NP.
Through reductions from classic NP-hard problems (such as 3-OCC-MAX2SAT and Vertex Cover), the paper rigorously establishes the intractability of tokenisation even under practical, real-world constraints.

### Strengths
The connection drawn between tokenisation and classical NP-hard problems is conceptually elegant and insightful.

The paper tackles a relevant and previously unresolved problem, making a clear theoretical contribution to both computational complexity and NLP.

The authors thoughtfully discuss implications for practical tokenisation algorithms, encouraging further research on approximation and relaxation methods.

### Weaknesses
The discussion of approximation hardness constants (e.g., very tight lower bounds) could be expanded to give more intuition on their practical implications.

### Questions
Could the authors clarify whether their hardness results extend to probabilistic or heuristic tokenisation schemes commonly used in NLP (e.g., BPE variants with stochastic merges)?

Do the reductions rely crucially on compression-based objectives, or might similar hardness results hold for alternative objectives like frequency balancing or entropy minimization?

Given the established hardness, do the authors foresee any provably efficient approximation algorithms for special cases (e.g., very small datasets or restricted merge operations)?

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
4

### Summary
This paper addresses the computational complexity of bottom-up and direct tokenisation over bounded alphabet. The authors develop elaborate proofs to show that this problem is NP-complete even with alphabets consisting only of 2 characters. Also they show that there is no polynomial time approximation unless P=NP. The main optimization function is compression. The paper contains elaborate formal definitions of the problems addressed and provides elaborate technical details underlying the main results.

### Strengths
The paper presents rigorous proofs for tokenisation over bounded alphabets with compression as the optimization function.

### Weaknesses
In my opinion, this paper will not be of interest to the main ICLR community and is only of theoretical interest with no clear practical impact on learning or NLP.

### Questions
How does this work fit under learning representation? 
There are already heuristics to perform tokenisation which work very well in practice with no theoretical guarantees. How does this work impact the existing vast literature?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors consider the problem of tokenization: given a database of words, how can one most efficiently tokenize the words? Tokenization is an important first step in most natural language processing pipelines, and finding the optimal tokenization is potentially useful to optimizing language model performance. Thus, it is important to understand how well we can hope to compute good tokenizations. Formally the authors consider two problems:

1. Direct encoding: given a set of tokens, the tokenization of a database is the optimal splitting of the words in the database to minimize the number of tokens used. 
2. Bottom-up encoding: given a set of tokens, the tokenization of a database is represented by a set of merge operations. The tokenized database is obtained by sequentially applying the merge operations greedily to the database.

For both formulations of the tokenization function, the optimization problem asks to find the tokenization (i.e. the set of tokens in the former case, the merge operations in the second case) minimizing the number of tokens. Naturally, one can also consider approximations of this problem. The main result of this work establishes that unless P = NP, there is no PTAS for tokenization i.e. there exists a constant 1.00001 such that finding a 1.00001-approximation to the optimal tokenization is NP-hard. The proofs follow via reduction to a special case of the MAX-2-SAT problem. Whereas previous work obtained lower bound for unbounded alphabets, this work gives the first hardness result for bounded alphabets, and in fact gives a strong result ruling out polynomial algorithms for binary alphabets. Furthermore, they show that for (1) even unary tokenization is NP-complete.

The authors consider an interesting problem, and give strong results. The paper is well written and easy to follow. I therefore recommend accept.

### Strengths
The paper studies a practically motivated problem which is a key step in training natural language processing models. The result is strong and essentially resolves the question of tokenization with compression as an objective. The paper is well written, and the proofs are well motivated and easy to follow.

### Weaknesses
No clear weaknesses (see minor comments below)

### Questions
Minor Comments 

Reference to Lemma 1 and 2 - I think better to reference Reduction 1 and Theorem 2

Maybe good to state explicitly what all merge operations are in forward step of bottom-up tokenization proof.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper investigates the tokenization problem, which is the first step in most NLP pipeline. The problem (somewhat involved to describe) is the following: Let $\Sigma$ be a finite alphabet  and let $D = \{c_1, c_2, \dots, c_M\}$ be a dataset of character strings 
$c_i \in \Sigma^{*}$. A *tokenizer* is a tuple $(S,dt,t)$ where $S$ is a *vocabulary* which is a finite set $S \subset \Sigma^{+}$ of nonempty substrings over $\Sigma$. $dt$ (detokenizer) is a function from $S^{\*}$ to $\Sigma^{\*}$ which is a string concatenation operator. The tokenizer encoder unction $t$ is a function from $\Sigma^{\*}$ to $S^{\*}$ such that  for any $c \in \Sigma^{\*}$, $c = \mathrm{concat}(t(c))$.  Fix a natural encoding function (eg direct encoding) and a budget $K$. For a given vocabulary $S$ so that $|S| = K+|\Sigma|$, the cost of $S$ is the minimum,  total number of tokens produced over all the dataset $D$ -- i.e., achieves maximal compression over $D$. The optimization problem is to find cost of the best-cost vocabulary of size $K+\Sigma$.

The paper considers two types of encodings -- direct encoding and bottom-up encoding, which are used in practice and investigated in the literature. Prior work has shown that these optimization problems are NP-hard when the alphabet size is unbounded. The present work builds on this hardness result and  strengthen them in several ways. They show that in both cases, even if the alphabet is binary, these problems remain NP-complete. More interestingly to me, they show that even approximating the optimum value up to arbitrary accuracy (PTAS) is NP-hard. In other words, there exists a constant $\varepsilon > 0$ such that no polynomial-time algorithm can approximate the optimum value within a factor of $(1+\varepsilon)$ unless P=NP. The paper leaves open a very interesting theoretical question, is there a constant ratio approximation algorithm for this problem? Concretely, is there an approximation algorithm that finds a vocabulary whose cost is  almost twice the optimum?

### Strengths
The computational problem studied in this paper is highly relevant, and, given the prior work, the demonstrated impossibility of approximation algorithms with arbitrary precision represents a significant theoretical contribution. The results provide valuable insights into the computational complexity of a fundamental step in modern AI and NLP models. The paper is clearly structured and written fairly well. While I did not verify all proofs in detail (as they are presented in the appendix), the claims appear sound and consistent with the constructions outlined in the main text. I also find the open question regarding the existence of a constant-factor approximation algorithm particularly intriguing—and I even suspect that the answer might turn out to be negative.

### Weaknesses
While theoretically it is an interesting paper, it appears like the practical tokenizers work very well and not clear whether these results will have any impact on the progress of modern NLP models. Another weakness I find is that it appears complicated than it needs to be to define the computational problems. Some notational use appears non-standard. For example while $tok$ is a function by definition, they also use it as a set and use notations such as $|tok|$ which, to my understanding is the cardinality of the vocabulary. I find such notional use a bit confusing. Another minor weakness is that, the paper is dense and proof-heavy with limited examples illustrating the reductions. Hence a small running example of the reduction would improve readability.

### Questions
From a theoretical viewpoint, It will be very nice to see the exact constant beyond which one cannot approximate. You give a bound 1.000002, which is theoretically sufficient for the claim, but practically a 1.000002 approximation algorithm is very good.  Have you considered improving the constant? What is your insight as to the existence of constant factor approximation algorithm?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper investigates the computational complexity of tokenisation when applied to inputs drawn from bounded alphabets, addressing a gap in earlier NP-completeness results that assumed alphabets of unbounded size. Tokenisation, a core component of natural language processing pipelines, converts character strings into subword sequences. Many tokenisation methods—such as Byte Pair Encoding (BPE) and Unigram Language Models—aim to compress data, thereby improving efficiency in model training and inference. While previous work had established that finding an optimally compressive tokeniser is NP-complete, those proofs relied on the unrealistic assumption of infinitely large alphabets.

The authors define and analyse two bounded-alphabet tokenisation problems: bottom-up tokenisation, where an optimal sequence of merge operations must be selected, and direct tokenisation, where the optimal vocabulary must be chosen directly. They demonstrate that even when the alphabet is extremely limited, the problems remain computationally intractable. Specifically, for binary alphabets (only two characters), both bottom-up and direct tokenisation are shown to be NP-complete and to lack any polynomial-time approximation scheme, unless P = NP. The authors also show that direct tokenisation remains NP-complete even for unary alphabets (containing a single symbol). Because unary and binary alphabets are the simplest cases, these hardness results automatically extend to all larger alphabets.

The findings imply that the difficulty of tokenisation does not stem from large alphabets or complex merge strategies, but from the inherent structure of the optimisation problem itself. Consequently, it is unlikely that any efficient algorithm can find or even closely approximate an optimal tokeniser under a compression objective.

### Strengths
The main strength of the paper is that it closes a gap in earlier NP-completeness results, which assumed alphabets of unbounded size. The current paper shows that these hardness results hold even for small size alphabets.

### Weaknesses
The scope of the paper may be more suitable for a conference on computational complexity. On the other hand the results are about an important problem in natural language processing. Therefore it may fit a section dedicated to computational complexity results within natural language processing.

### Questions
No question

### Soundness
3

### Presentation
3

### Contribution
3
