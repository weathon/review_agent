# PSC: Efficient Grammar-Constrained Decoding via Parser Stack Classification

- Decision: Reject
- Scores: 2, 4, 2, 6

## Abstract
LLMs are widely used to generate structured output like source code or JSON. Grammar-constrained decoding (GCD) can guarantee the syntactic validity of the generated output, by masking out tokens that violate rules specified by a context-free grammar. However, the online computational overhead of existing GCD methods, with latency typically scaling linearly with vocabulary size, limits the throughput of LLMs, especially for models with large vocabularies. To address this issue, we propose PSC, a novel grammar-constrained decoding method. By combining acceptance conditions of all vocabulary tokens into a single classifier of the parser stack during preprocessing, PSC can compute the complete vocabulary mask by checking the parser stack exactly once per decoding step, with time complexity independent of the vocabulary size. Experiments show that PSC computes masks up to 770× faster than baselines on complex programming language grammars, and up to 30× faster for schema-conformant JSON; end-to-end LLM throughput with PSC approaches that of unconstrained decoding.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a new approach for speeding up grammar constrained decoding. 
The key insight is a clever one, instead of "running" the parser alongside with the decoder, one only needs to understand where in the parser the decoder in terms of stack state, and there are only finitely many "classes" of stack states that are interesting in terms of masking tokens away to perform constrained decoding. Furthermore, as known from theory of computation, these equivalence classes are recognizable by regular languages over stack traces.
One can therefore precompute all masking and do them very efficiently at decoding time.
The paper shows how this approach incredibly fast in practice (less than 3 microseconds overhead per token), thus defeating all the state of the art approaches.
The results only include speed of decoding, but not the so called "time-to-first-token" and is therefore not know what the pre-processing time of the tool.

### Strengths
- Clever idea of only considering regular languages of stack traces so one can avoid computing masks at runtime
- Evaluation shows clear gain over state of the art in terms of speed and in a way sets an unbeatable bar for future approaches.

### Weaknesses
The paper has two main problems, one fixable and one major.

Fixable problem: 
The exposition of the paper is really really hard to follow. I knew the algorithm after the abstract (probably because I had thought about the same idea before, though I didn't expect it would have such a strong impact in time), and yet I couldn't follow the presentation. 
Many symbols are undefined (e.g., I had to guess that \Pi is the set of stack symbols), the notion of "stabilized version" and stable stack are never defined, I don't know what {Final} means.
Many references to theorems and definitions are incorrect within the paper, making me guess where to read most of the time. Theorem 1 mentions Definition 4, but there are no definitions (so I assume it means equations). Similar issue with theorem 2. Line 274 refers to "simplification 3" but again, maybe it means equation and same for Definition 1 one line before. The algorithm are sort of given "as is" without an example of the intuition behind them. I would remove all the algorithms and put them in the appendix and focus the presentation on showing an example.

Major problem: 
Now comes what I think is big limitation of the paper, it "hides" the cost of preprocessing. I'm 99% sure that this approach has very very high preprocessing time. Yet, I would be ok with it if it wasn't that the paper decided to hide this aspect and just say briefly in the conclusion "so the overhead of preprocessing is generally acceptable for practical applications". The tools the authors cite and compare against LLGuidance and GreatGramma ALL report time to first token (aka preprocessing) and time per token. So it is clear that people care about both of these quantities. In fact, GreatGrammar is explicitly built to balance the tradeoff between these two quantities. Without reporting preprocessing times, this paper CANNOT be accepted. On a related note, the authors should discuss the computational complexity of the approach.

There are some applications where preprocessing may not matter because the grammar is set once and for all at the beginning, but in most cases people use the grammar to "program" the output of the LLM and often do so at running time (e.g., they specialize the grammar to certain variable names or namespaces).

### Questions
What is the preprocessing time of PSC on all the grammars discussed in the paper and how does it compare to other tools?
(Once I see these numbers, I can form an opinion on whether I should update my score)

What is the computational complexity of the construction in PSC?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes PSC, a GCD method that precomputes a finite-state automaton mapping parser stacks to valid tokens, reducing decoding time complexity.

### Strengths
The paper focuses on improving the efficiency of grammar-constrained decoding, and the writing is clear and easy to follow.

### Weaknesses
Novelty clarification: The core idea of PSC, precomputing finite-state representations of grammar validity, is well established in compiler theory and automata-based parsing. The paper would benefit from clarifying what is genuinely new about PSC beyond these known concepts. Also in practice, token masking is often not the main bottleneck compared to the model’s forward pass, so the real-world efficiency gains may be less pronounced.

Distributional bias and downstream performance: The paper focuses on decoding efficiency but does not examine whether enforcing strict grammatical validity alters the model’s output distribution and therefore degrades the downstream task performance. GCD is known to introduce such biases [1], so evaluating downstream quality, e.g., code accuracy, semantic correctness, or task performance, would make the results more complete.

Practical limitations: It would be helpful to discuss the preprocessing time and memory footprint of building and minimizing the automaton (reporting the total cost of the entire process rather than only the decoding phase), as well as how PSC handles dynamically changing grammars or schema updates. Quantifying these trade-offs would improve the paper’s transparency and practical relevance.

[1] https://arxiv.org/abs/2405.21047

### Questions
See above weakness.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper aims to reduce the processing time of grammar-constrained decoding (GCD). The authors argue that existing approaches must check every token in the vocabulary against the parser to ensure syntactic validity, leading to an O(|V|) runtime complexity—though this claim is somewhat overstated, as many prior methods already exploit token tries or sparse validity masks. The proposed method, Parser Stack Classification (PSC), precomputes the results of the lexing and parsing process for all possible combinations of parser states and tokens (or terminal sequences). At runtime, PSC avoids re-running the parser and instead retrieves precomputed validity results from a large lookup structure, achieving up to 770× speedup in mask computation and significant throughput gains.

The core contribution is not conceptual novelty but rather a proof-of-extremes: the paper demonstrates that, if one is willing to precompute and store enough information, the runtime cost of grammar checking can be minimized almost completely

### Strengths
The paper tackles a practically relevant problem — reducing the runtime cost of grammar-constrained decoding. The authors demonstrate clear effort in building a formal framework (although overstretched and overcomplicated) that connects the lexer, parser, and grammar validity checking. The underlying idea of trading runtime complexity for offline precomputation is conceptually reasonable and aligns with longstanding efficiency strategies in compiler and parsing theory. However, while the overall idea is sound the approach leans too heavily on exhaustive precomputation to claim constant-time decoding. Still, the paper's motivation, completeness of implementation, and inclusion of multiple baselines reflect solid execution effort.

### Weaknesses
1. Computation is not eliminated, only shifted to preprocessing.
The paper's central claim—that PSC achieves "runtime independent of |V|"—is misleading. The method still performs preprocessing for every token and ultimately stores a full |V|-sized mask for each combined lexer–parser state. In effect, it trades O(|V|) runtime cost for O(|V|) offline computation and storage. This does not save computation—it merely moves it earlier in the pipeline. Moreover, many of the precomputed state transitions will never occur at runtime, making the approach less efficient overall. This is conceptually similar to precomputing all n×n matrix multiplications in order to claim O(1) inference—a theoretically valid but practically meaningless optimization.
The paper also provides no quantitative discussion of precomputation time or memory footprint, despite these being the dominant costs. Given that the automaton A stores a |V|-bit mask per stack-equivalence class, the storage requirements for nontrivial grammars can explode rapidly. Without measurements of preprocessing time, number of automaton states, or disk/memory usage, the practicality of PSC remains unsubstantiated.

Finally, the authors' criticism of prior work ("existing methods compute over all tokens") is not accurate. Many grammar-constrained decoding systems already use trie-based lexers or incremental constraint propagation, achieving complexity proportional to the number of valid tokens rather than the full vocabulary. PSC's precomputation does not strictly improve upon these optimizations and may in fact be less efficient when constraints are sparse.

⸻

2. Low novelty relative to existing approaches.
Several prior GCD systems (e.g., XGrammar, GreatGramma, Formatron) already perform offline caching of terminal sequences, context-independent masks, or parser states. The claim that "previous approaches cannot fundamentally change O(|V|)" is true yet not really meaningful, given that many already reduce effective runtime complexity through structured token tries or hierarchical lexers. Overall, the idea of moving computation all to preprocessing is limited in novelty and not creative.

⸻

3. Overcomplicated and low-yield theoretical presentation.
Part of the paper's exposition is unnecessarily formal. Sections 3.3–3.5 introduce multiple "theorems" that are self-evident ("by construction") and add little technical insight, while essential notation (e.g., Π for the stack alphabet) is deferred to the appendix. The presentation is cluttered with symbolic machinery—ε-FSTs, multi-level DFAs, nested composition operators—that obscure a relatively simple idea: precompute parser transitions and look up validity masks at runtime. Compared with prior work such as XGrammar, the writing here is substantially less readable and less disciplined in its abstraction boundaries. Simplifying the derivations and focusing on empirical scalability would make the contribution far clearer and more impactful.

——

While the paper's motivation—achieving faster grammar-constrained decoding—is clear, the work is fundamentally limited by its blind pursuit of "runtime efficiency." PSC achieves constant-time decoding only by offloading essentially all computation into an enormous precomputation phase. This tradeoff is not just theoretically questionable but also practically unappealing: it replaces a manageable online cost with unbounded preprocessing time and memory demands. Although such precomputation is technically possible, it is neither interesting from a research perspective nor useful in real deployments. The authors should provide a much deeper discussion on the balance between runtime efficiency and pre-runtime cost, including when such a tradeoff is actually worthwhile and how the claimed runtime benefit compares quantitatively to the enormous precomputation overhead. A convincing evaluation would require detailed reporting of:

- **Wall-clock precomputation time:** how long it takes to build P_\varepsilon, \tilde P_a, all P_w, and the final automaton A.
- **Automaton size:** number of states |A| before and after minimization.
- **Disk or RAM usage:** total bytes of transition tables or per-state vocabulary masks.
- **Scaling curves:** how preprocessing scales with grammar complexity or vocabulary size (|Γ|, |Π|, |V|).
- **Compression results:** any mention of FSA minimization effectiveness or disk savings.

And the above is certainly grammar dependent so the reporting should also span a variety of grammar in terms of the number of parser and stack states.

### Questions
See weaknesses

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper presents an efficient grammar-constrained LLM generation technique called PSC. Given any partial generation PSC computes the set of acceptable tokens according to the grammar in following steps. For each token, PSC precomputes a FSA that represents the condition for parser state to accept the token. These FSAs are combined into single FSA that maps parser state to vocabulary masks. 

During decoding, a lexer is used to compute lexical tokens corresponding to the partial generation. This is used to compute the parser state. Using the encoded mapping through the FSA, a vocabulary mask is obtained efficiently by a simple lookup. The evaluation results show that - while small language models constraining on grammars for JSON, Java, Go, SQL generation PSC outperforms prior state-of-the-art techniques such as Xgrammar, Formatron, GreatGramma and LLGuidance in terms of efficiency.

### Strengths
1) There have been several grammar-constrained generation works in recent years. The grammar-constrained generation techniques have been commonly used for function-calling and it has become standard to support this feature on all LLM serving engines. PSCs throughput results are impressive and would likely help in further adoption of constrained-generation techniques for other more complex programming language grammars. 

2) The presentation is detailed and the guarantees provided by the technique are proved rigorously.   

3) The throughput experiments consider state-of-the-art baselines such as Xgrammar, Formatron, GreatGramma and LLGuidance and shows improvement over them. 

Overall, I’m positive about the paper and happy to increase my score further if the authors could address my remaining concerns.

### Weaknesses
1) The paper does not report: (1) offline precomputation time for each grammar, (2) memory footprint of the combined FSAs, (3) how these scale with grammar complexity


2) Some of the claims about all prior works requiring $O(|V|)$ parsing steps in the worst case are inaccurate.

> However, none of these techniques can fundamentally change the time complexity, which is still $O(|V|)$ in the worst case.

Syncode (Ugare et. al.) performs single parsing step per decoding step as well. Syncode also computes set of acceptable terminal sequences (using LR parser) referred as accept sequences which is similar to realizable terminal sequences in PSC. 

The main difference appears to be that Syncode requires mask lookup for each terminal sequence and a union operation over those masks, while PSC combines FSAs offline leading to a single lookup during inference. Can the authors confirm this characterization is accurate?

3) While PSC maintains syntactic correctness guarantees (Table 2), it's unclear whether the efficiency gains allow for practical improvements in downstream tasks.  Even if PSC does not lead to improved semantic correctness, I would encourage the authors to include an experiment for computing pass@k accuracy on standard code generation tasks. 
 
Nits:

* Section 4.2 should mention the size of models used for the experiment  
* I spotted a few places with missing whitespace after punctuations. Line 149, 482. 

> Line 194: While some methods employ precomputation to optimize certain cases, they still fundamentally require

Precise citations here will be helpful.

### Questions
q1) Is the combination of FSAs the primary novel algorithmic contribution compared to GreatGemma and Syncode? Can you provide an ablation study focusing on efficiency gains from: algorithmic improvements such as FSA combination to disregard the gains from implementation optimizations like use Cython? 

q2) What is the effect of lookahead of LALR parser in overall strength of PSC? Are there any downsides in using LALR parser instead of an Earley parser?  

q3) What is the offline time taken for pre-computation performed by PSC for grammars considered in the evaluation? 

q4) What is the memory footprint of the combined FSA structures for each grammar?

### Soundness
3

### Presentation
3

### Contribution
3
