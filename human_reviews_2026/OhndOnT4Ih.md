# Constrained Adaptive Rejection Sampling

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 2, 8

## Abstract
Language Models (LMs) are increasingly used in applications where generated outputs must satisfy strict semantic or syntactic constraints. Although constrained decoding methods can enforce constrain satisfaction, they often alter the underlying LM distribution, limiting their usefulness in settings such as program fuzzing, where both validity and diversity of samples are essential.
We present Constrained Adaptive Rejection Sampling (CARS), a simple yet principled alternative to constrained decoding that enforces constraints without distributional distortion. CARS begins with unconstrained LM sampling and adaptively rules out constraint-violating continuations by recording them in a trie and subtracting their probability mass from future draws. This adaptive pruning ensures that prefixes that have been proven invalid are never revisited, acceptance rates improve monotonically, and the resulting samples exactly follow the constrained distribution. In experiments on synthetic domains and program fuzzing benchmarks from prior work on constrained decoding, CARS consistently achieves higher efficiency and stronger sample diversity compared to prior constrained decoding techniques.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper presents Constrained Adaptive Rejection Sampling (CARS) algorithm for constrained LLM generation. 
It extends existing Adaptive Rejection Sampling (ARS) for constrained decoding. CARS adaptively removes the continuations that may violate the constraints and reweighs the probability for the remaining paths. The paper adaptively  intercepts constraint-violating continuations, stores them and subtracts their probability mass from future draws. The paper shows their approach on three tasks, program fuzzing, molecular generation, and PDDL planning.

### Strengths
The paper expands the reach of rejection sampling in constrained LLM generation with a new algorithm for sampling that is aware of grammatical constraints. The running example is helpful.

The paper presents the proof that the CARS algorithm produces unbiased samples from the true constrained distribution and the sample-acceptance rate increases monotonically

CARS is evaluated on multiple scenarios and against multiple baselines. The evaluation of the metrics related to the number of samples is detailed and the appendix contains additional results and interpretations.

### Weaknesses
The technical contribution (Algorithm 5) is relatively straightforward and the technical novelty over existing methods is low. The main novelty is the update strategy described on lines 235-242

The reshaping of the distribution R^W seems to be a costly operation. Doing it in each iteration may increase costs substantially. There is of course, a tension between the number of LLM cals and sampling time, but this should be clearly characterized. Further, the trie data structure may grow substantially for larger problems (and its search/insertion complexity is not trivial). Both issues raise some doubts about practicality of the proposed approach. The evaluation should  show the execution time and memory consumed compared to other baseline methods and how well they scale over generation time.

The experimental results  do not consistently show substantial utility for two out of three studies:

 * Fuzzing / XML: while CARS shows improvement in coverage over the approximate methods (while it is reported ARS does not produce satisfying samples). However, the base coverage for all those methods is fairly small (9%), and final improvement is marginal. More importantly, such a low rate of code coverage does not give much confidence for correctness of tested code (without a more qualified discussion it may be that the grammar is testing only a very small part of the language). 

  * Fuzzing / JSON and SQL (appendix): they do not show a clear improvement of CARS over RSFT, even for sample efficiency on Qwen w/ grammar. Moreover, the fuzzing paper that the authors citedm the branch coverage over 30% for SQL using the MCMC generation methods (Fig. 3a in Gonzalez et al NeurIPS 2025), indicating a potential problem with the choice of the simulation length in the experimental setup. 

  * Molecular generation: the CARS results do not show significant improvement over ARS and RSFT. The results from the appendix show that the two are equal, or ARS can even be better in some scenarios. The number of samples is also similar in the majority of the cases. 

The evaluation baselines should include ASAp (Park et al. 2024) as a related state-of-the-art baseline for approximate generation. The existing description in the related work is insufficient to get the full understanding of the advantages and disadvantages in practical settings. Interestingly, the paper mentions ASAp at one point in the evaluation and the appendix H uses its benchmarks, but the experiments do not use it.

### Questions
Please discuss the concerns raised regarding the cost of reshaping/trie and comparison to ASAp.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents Constrained Adaptive Rejection Sampling (CARS) method that adresses the efficiency-fidelity tradeoff in constrained language model generation. CARS maintains a trie data structure of constraint-violating prefixes and uses it to adaptively prune invalid paths during sampling. This approach offers two key advantages: it achieves exact distribution matching, unlike greedy constrained decoding methods, while significantly improving sample efficiency compared to standard rejection sampling.

### Strengths
- The paper is generally well-written, especially with the use of illustrative example.
- Constrained LLM sampling is a crucial problem. Hence, a contribution that improves constrained sampling can have significant impact on practical applications in code generation/fuzzing/molecular synthesis.
- The CARS algorithm provides theoretical exact convergence guarantee.

### Weaknesses
- The main technical section of the paper should describe the technical contributions of the paper in comparison to the prior works on adaptive rejection sampling [1] and MCMC [2]. These works have been considered in the evaluation. However, the technical section lacks proper comparison to these techniques. After reading this section I still cannot tell exactly what is the main technical novelty of the CARS algorithm in comparison to these works.  

- There is small (~0.3 pp) improvement in the branch coverage improvement compared to MCMC for the fuzzer on XML evaluation. In the PDLL evaluation, there is no or insignificant improvement in accuracy.  Overall, none of the experiments show significant improvements over prior works. I recommend authors consider tasks such as text-2-sql generation or code generation tasks considered in prior constrained LLM generation works.

- The evaluation only considers two small non-SOTA models (Qwen2.5-7B-Instruct, Llama-3.1-8B). The authors should include more models and show that the technique generalizes to other models. 

## Minor

Line 24: Abbreviation GCD is not defined till this part

Table 2: The Table bolds results that are worse in prefix validity

[1] Fast Controlled Generation from Language Models with Adaptive Weighted Rejection Sampling. Lipkin et. Al.

[2] Constrained sampling for language models should be easy: An MCMC perspective. Gonzalez et. al.

### Questions
What is the practical motivation for exact distribution sampling compared to existing asymptotic convergence guarantees provided by the prior works?

Lipkin et. al. [1] show text-to-sql experiment for showing improved accuracy on SQL generation. Can you compare CARS with other baselines on this task? Given that none of the techniques show good accuracy on the PDLL tasks, I think text-to-sql task would be a better evaluation for the work.

### Soundness
2

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
This paper presents CARS (Constrained Adaptive Rejection Sampling), which improves upon standard rejection sampling for constrained language model generation by maintaining a trie of invalid prefixes and adaptively reweighting the sampling distribution. When generating constrained outputs (e.g., valid programs, molecules), CARS records not just rejected samples but all constraint-violating continuations of their prefixes, preventing revisitation and monotonically

### Strengths
- Theorem 1 is correct, and the proof sketch is sound
- The paper is clear and easy to follow
- Comprehensive ablations

### Weaknesses
- Lack of technical novelty. CARS is a straightforward extension of ARS. Furthermore, the approach highly resembles and lacks comparison to IterGen (ICLR 2025) [1]
- Lack of extensive model analysis, as only two models are used in the experiments.
- Lack of both qualitative and quantitative computational complexity analysis for costs such as maintaining the trie.
- The theoretical analysis seems trivial based on the problem and the algorithm design

[1] Ugare et al. IterGen: Iterative Semantic-aware Structured LLM Generation with Backtracking. ICLR 2025

### Questions
See Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
[Note: I wrote this review almost 2 weeks ago, but only noticed at the start of the discussion period started that I hadn't submitted it properly.  Apologies to the AC and the authors.]

The paper provides new exact methods for constrained sampling from an autoregressive language model.  It assumes (line 111) that constraint violation can be detected partway through a generated string, by detecting an invalid prefix.  

The Adaptive Rejection Sampling (ARS) method is basically just rejection sampling, but it gradually patches the language model to rule out all of the invalid prefixes that have been encountered so far.  The patched model can be used not only for retries during rejection sampling, but for future IID samples as well.

The Constrained Adaptive Rejection Sampling (CARS) method is a cute extension that does "constraint overgeneration" (one might say).  For each drawn string $w$ (whether accepted or rejected), it generates constraints that were "almost violated" -- all constraints of the form $\neg(ua\Sigma^*)$ where $u$ is a prefix of $w$.  

The methods are tested on 3 domains and appear to substantially beat competitors.  The tasks are drawn from previous work and the comparisons seem methodologically fine, though perhaps I should scrutinize the experimental design and results more closely.

There is substantial supplementary material, which I have mainly not reviewed but would be willing to look at.

### Strengths
The paper was a pleasure to read (especially compared to my other assignments).  Thanks!

The method is simple, but I consider this to be a virtue.  It would be easy to teach in an LLM class as a correct constrained decoding method.

Sections 4.2-4.3 show good empirical improvements over AWRS (Lipkin et al. 2025), which just won an award at COLM.  For this reason I think the paper is of current interest.  It's nice to see that exact sampling is not only feasible but can also be more efficient than approximate sampling (such as AWRS).

CARS is an attractive extension to ARS because it is computationally cheap in the common case where the violation checker, for every valid prefix $u$, aggressively computes the whole set of valid next symbols (so $a$ should range over all symbols *not* in that set).  For example, the violation checker may be running an FSA to check membership in a regular language, or Earley's algorithm to check membership in a context-free language.  I think this should be discussed explicitly.

### Weaknesses
No notable weaknesses. 

Minor suggestions:

[I will write \$ below as # because the Markdown parser gets confused when I use the former twice in the same paragraph.]

I think the equation at line 176 is buggy because it doesn't mention $\mathcal{W}$.  Presumably you mean tot restrict the summation.

You might consider an alternative presentation where the trie node for $u$ stores not $p_u$ but $r_u = 1-p_u$, that is, the total probability mass that the current $\mathcal{W}$ *removes* from the language model for continuations of $u$.  This might be clearer because then the trie is a more familiar object -- a restriction of the LM to illegal strings, inheriting its edge weights from the LM.  $r_u$ is just the total weight of all paths from $u$ to a leaf, and in particular $r_u=1$ when $u$ is a leaf (representing all strings starting with $u$).  Thus, each internal trie node stores a weighted sum of its (hopefully small) set of children, just as in the backward algorithm, with no subtraction within the trie.   The only subtraction would happen in the formula at line 204, when $p_u$ is computed as $1-r_u$.  

I think there are some inconsistencies about whether complete strings are said to *include* # or be *followed by* #.  At line 102 and elsewhere you require a complete string $w$ to include the final #, but at line 099 you require $w \in \Sigma^*$ which seems to incorrectly exclude complete strings from the $P(w | u)$ notation.  At line 150, you omit the # that is called for by the formalism; instead you could write E ::= d# | d + E (and include # in the example strings from this grammar, e.g., at line 241).  

I also would drop the length limit that you mention at lines 101 and 104 but sometimes ignore elsewhere.  You don't have to "artificially stop generation"; just modify the LM to generate # with probability 1 at token $N+1$, and then you'll satisfy the condition at line 102.

Under line 161, say that the function $p_{\cdot}$ is updated whenever $\mathcal{W}$ is.  In fact they're represented jointly in the same data structure.

At line 192, "the above formula" should have a formula number: I think you mean either line 161 or a recursive application of line 176 (though you don't implement 176 literally, instead starting at 1 and gradually subtracting).

RSFT at line 225 is a bit hard to follow.  I think you mean that it's a version of ARS where you limit to invalid prefixes of length 1.

Line 241 could be made more precise, since "every shorter prefix u" in (ii) is not well-defined in this case (shorter than what?).  I think you mean "every proper prefix u of w" (where w ends with #). 

You should discuss when CARS is efficient (see my comments about this elsewhere in the review).

### Questions
# Prior work

I think ARS is morally speaking an application of [Tromble and Eisner (2006)](https://aclanthology.org/N06-1054/), which similarly patches a language model with constraints as they are discovered to be violated.  Is that correct?  I do see three differences:

1. Tromble and Eisner are doing argmax decoding rather than sampling (though their method appears to apply equally well to sampling).

2. Tromble and Eisner use finitely many constraints, which are regular languages (FSAs).  The submission implicitly uses infinitely many prefix constraints of the form $\neg(u\Sigma^*)$ where the prefix $u$ is a fixed string, which is possible because it discovers them only as they are violated.  (The term of art is "[dynamic] constraint generation" or "row generation," the most famous example being subtour elimination constraints in the Traveling Salesperson Problem.)

3. Tromble and Eisner is a pre-neural paper and assumes that the language model is a weighted FSA.  The submission uses any autoregressive LM, so it replaces this FSA with the implicit infinite weighted trie corresponding to that LM (a deterministic *infinite*-state automaton).  The $\mathcal{W}$ trie in the submission is the result of intersecting that object with the hard prefix constraints that have been generated so far.  That yields a materialized finite trie in which all explicit edges have weight 0, but whose missing edges implicitly fall back to the language model.  

Similarly, are there precedents in the literature for the "constraint overgeneration" in your CARS method?  It seems like the sort of thing that must have shown up before in constraint generation methods. 

In your discussion of related work, you focus on your own setting of hard constraints.  But can't many of the methods in section 5 also handle soft constraints, e.g, the product of an LLM with some other factors, followed by global renormalization as in energy-based modeling?  (I think this should be acknowledged.)

Since AWRS is also based on ARS, can you give a discussion of the differences (and also incorporate it into the final version)?

# Efficiency

For CARS, how about the case where $u$ can be legally followed by only a few symbols from a large vocabulary (e.g. the outgoing arcs from the constraint DFA state reached by $u$)?  In that case, instead of having the trie node for $u$ store all the *illegal* next symbols $a$ (each such edge leading to a leaf $ua$ with $p_{ua}=0$), wouldn't it be cheaper to store the small set of *legal* symbols?

The "sampling efficiency" metric (line 280) omits maintenance overhead costs such as adding all of the illegal next symbols to the trie.  Are these costs negligible in practice?  

How did you implement the decoder and is the implementation available?  It seems that to put this method into production, you would want to integrate with vLLM so that the decoder keeps track of its position in the trie and reduces the softmax probabilities as described at line 204.

# Experiments

When drawing multiple samples, how does your acceptance rate improve over time (as $\mathcal{W}$ grows)?  It seems that the time to first sample might be long but then the work is amortized over further samples (in contrast to the other methods, I think).  Thus, the metrics that you report depend on your choice of 100 as the number of acceptances you need at line 371 and line 411.  If you needed only 1 valid sample, or 10000, how would the comparison change?

Can you give some qualitative discussion of why your methods worked well (or badly)?  What constraints are learned by $\mathcal{W}$, how much probability $1-p_\epsilon$ do they remove from the LM, and how quickly are they learned?

### Soundness
4

### Presentation
4

### Contribution
4
