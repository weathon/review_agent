# Automata Learning and Identification of the Support of Language Models

- Avg Score: 7.00
- Decision: Accept (Poster)
- Scores: 8, 6, 6, 8

## Abstract
We study the learnability of languages in the *Next Symbol Prediction* (NSP) setting, where a learner receives only positive examples from a language together with, for every prefix, (i) whether the prefix itself is in the language and (ii) which next symbols can lead to an accepting string. This setting has been used in prior work to empirically analyze neural sequence models, and additionally, we observe that efficient algorithms for the NSP setting can be used to learn the (truncated) support of language models. We first show that the class of DFAs with at most $n$ states is identifiable from positive examples augmented with these NSP labels. Nevertheless, even with this richer supervision, we show that PAC-learning DFAs remains computationally hard, and exact identification using only membership queries cannot be achieved in polynomial time. We then present $\mathrm{L_{nsp}^{\star}}$, an extension of Angluin’s $\mathrm{L}^{\star}$ algorithm, and show that DFAs can be PAC-learned efficiently using a language-model–based teacher that answers membership queries and generates valid strings conditioned on prefix prompts. Finally, we conduct a comprehensive experimental evaluation on 11 regular languages of varying complexity. Using $\mathrm{L}^{\star}_{\text{nsp}}$, we extract DFAs from Transformer-based language models trained on regular languages to evaluate the algorithm’s effectiveness and identify erroneous examples.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes a next-symbol-prediction-based formal language learning framework with applications in extracting formal descriptions of neural language models. This framework augments Gold’s seminal language learning framework with information about the allowed string continuations. This allows much larger classes of languages to be learnable from positive examples alone. Nevertheless, the authors show that the added information still does not enable efficient PAC learning of finite-state automata (FSAs), but in some cases allows for more efficient L*-style learning based on membership and equivalence queries. 
The L* learning framework is compatible with using a language model as an oracle, which enables the authors to use the introduced algorithms to extract FSAs from trained language models. They find the algorithm to be very sample-efficient. Moreover, the extracted FSAs often agree with the ground-truth (whenever the neural model is well-trained), unless the language is hard to learn (like parity), as expected.

### Strengths
- The paper is very well structured and clearly written. The contributions are clear and in my opinion important.
	- Most significantly, the paper takes a well-studied and known learning setting and updates it to a well-motivated and relevant setting useful in modern language models
- The results are theoretically thoroughly backed and explained with well-composed proofs.
- Although the main contributions are theoretical, the paper provides extensive and well-justified empirical validation of the results with languages well-studied in the theoretical literature on neural networks and formal languages.
	- These experiments show the practical improvements of the proposed algorithm, which is also theoretically shown to be much more efficient in some cases
- I found the FAQ section to be useful. 
- It is useful that the algorithm is completely specified and described understandably in pseudocode.

### Weaknesses
- The work is very well written and presented. I found no major weaknesses.
- I just had one thought: I imagine looking at automata learned from some actual large-ish-sized language models would be interesting—maybe specifically domain-specific or domain-prompted large LMs. I imagine this is not possible with the current algorithmic complexities?

### Questions
- How does one (gracefully) handle learning states that are technically learned by the language model but not in the truncated support? 
- How does one choose the truncation threshold for the support of a language model?
- Any intuition of what happens when the language is not actually regular? The number of states just keeps increasing?
- Are there any convenient descriptions of languages where NSP information provably makes learning more efficient (even though it, in general, does not hold)?
- I believe that, technically, it can hold that a symbol can be in the set of permissible continuations in a neural LM but never lead to a string in the support if the generation never halts (so-called improper or non-tight language models). Is that intuition correct? If so, should this condition be enforced? I suppose this is equivalent to enforcing a finite expected length, which you do?
- Two nit-picks:
	- Is the framing of “positive examples only” completely justified? As mentioned, each (positive) example also brings with it possibly many negative examples (prefixes). This is purely a naming thing.
	- If the language model is regular (as assumed), the distribution over negative examples should also be regular (and thus defined and available)

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper formalizes support identification for language models in an NSP supervision setting that mirrors LM decoding, establishes identifiability of minimal DFAs from positive NSP data, proves worst‑case hardness results that survive NSP labels, and proposes $L^\*_{nsp}$, an extension of Angluin’s $L^\*$ that uses LM‑simulable queries (membership and prefix‑conditional generation). Experiments with Transformer teachers on 11 regular languages demonstrate that modest positive NSP data suffice to recover the target or teacher‑support DFAs and to detect erroneous strings when teachers deviate.

### Strengths
1. NSP exactly captures what a practitioner can extract from an LM under top‑k/top‑p/min‑p decoding. The focus on support (rather than probabilities) is natural and practically important.
2. $L^\*_{nsp}$’s Lemma 5.1 is a neat reduction from continuation mismatches to standard counterexamples using at most one generative query. This method does not require a perfect equivalence oracle which is often infeasible in practical.
3. The evaluations are careful and complete. The method recovers large Dyck DFAs with tens–hundreds of positives and surfaces nontrivial errors. The ablations support the claim that continuation labels are particularly beneficial on languages with dead‑state transitions.

### Weaknesses
1. The regular-language focus is reasonable for support identification, but large vocabularies are the key obstacle for LM-scale studies. While the paper is honest about this, many NLP applications involve **huge vocabularies**. The closure step’s $(O(|Q||T||\Sigma|))$ cost is problematic for realistic LMs, and the paper does not yet provide system‑level accelerations. If the authors can solve this concern I am glad to raise my score.
2. The PAC guarantee is **with respect to the teacher LM’s own distribution**. This is appropriate for support identification but complicates cross‑teacher comparisons and may hide rare but catastrophic support errors (if the teacher rarely visits those prefixes). (Section 5.) 
3. $L^\*_{nsp}$ sometimes needs one **generative** query to produce an accepting suffix. If the teacher’s support is not regular or exhibits long nonterminating paths under a truncation rule, the procedure may stall (H.3).

### Questions
1. How sensitive is the learned support DFA to the choice and hyperparameters of truncation (top‑k vs. top‑p vs. min‑p)? Could the same teacher yield meaningfully different supports across rules?
2. If the true support is not regular but “close,” does $L^\*_{nsp}$ converge to a minimal DFA that best approximates NSP labels in expectation (e.g., a smallest consistent DFA on observed NSP data)?

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
5

### Summary
This paper introduces a new model for automata learning where the queries the learner can ask are the ones that are more naturally provided by a language model that exposes token logits at each step. That is, in normal automata learning, one can only ask membership queries (is this string in the language) and equivalence queries (did I find the right automaton, if not give me a counterex). Whereas the authors propose a setting where one can, given a sequence a1...an ask for every token b, whether a1..aib is in the language. This is a very powerful oracle that can provide many positive and negative examples at once. The paper shows that even with this powerful new model identification in the limit remains hard. They therefore move to a setting where the learner can ask the LM itself to generate positive sequences (together with their possible continuations). This approach yields a PAC guarantee (as long as equivalence is checked on the same distribution of the LM). The empirical analysis shows small DFAs can be relearned from a transformer (that were trained to recognize them).

### Strengths
- The formalization of NSP queries is a clean one.
- The paper studies a question worth studying and though the hardness result is somewhat unsurprising (and also the generative one since very similar results hold for learning DFAs when strings are sampled from a distribution to mimic an equivalence oracle), it is nonetheless an interesting result that NSP queries help learning faster empirically as shown (i.e., if one has the examples handy, might as well use them).
- The paper is formal and self-contained

### Weaknesses
- My main criticism of the paper is that open source LLMs provide the perfect setting for evaluating interesting questions and this paper instead has gone for training small transformers on toy languages. Before reading the eval description in the introduction, I was expecting this paper to do the following
1. Prompt the LLM (any open one) with the query "Generate strings that are accepted by the regular expression R"
2. Use L*NSP to learn what the LLM generate
3. Compare automaton to R
Then the evaluation would tell us whether 2B parameter models learn worse than 8B parameter models, for example, and what type of mistakes they make. I would love to see such an evaluation. One really interesting aspect would be to see when using (Real-world regexes) what characters LM "prefer" within character classes and how they fail to cover them
- Related to the above problem is that the paper does not seem to discuss the fact that LLMs produce tokens and not characters.

### Questions
Can the authors do the experiment I discuss in weakness for some interesting regular expressions and small open LLM? Ideally with some practical regexes than Dick languages (maybe pick 10 from regexlib or automatark)

How does the algorithm change when moving from characters to tokens?

Why are there EOS tokesn between characters in 393? IS this a way to force a certain tokenization scheme?

MINOR COMMENTS:
- line 145: missing _L in phi
- what does well-defined mean on l 210?
- Spell out PDFA in line 388 and point to a reference
- is the reference to Sec 2.2 in line 400 incorrect?
- maybe relevant paper to cite https://arxiv.org/abs/2501.02825

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
3

### Summary
The paper is a theoretical and empirical contribution to understanding language models learnability of formal languages by means of extracting automata from them. The data is regular languages with NSP labels (given a prefix, its acceptance as a prefix and the possible next symbols are labeled). The work gives a formal proof that for a given sized DFA the machine is identifiable with strings annotated with the NSP labels. The work also gives a L* variant (classic algorithm for identifying automata) algorithm for identifying the automaton by making use of a learned membership query and a simulated next symbol prediction, both using a trained language model (as in the original L* but with an oracle). A PAC guarantee is given that is connected to the error of the language model. This allows for extraction of DFAs from trained models, and can serve as a way to understand what “automaton” the trained model is simulating given the training data. Such an empirical study is conducted using 11 regular languages. There is also a hardness result showing how membership alone does not suffice to learn.

### Strengths
The work is thorough and shows us how we can cleverly extract automata from language models trained on synthetic data. This is backed up by theoretical results. The problem is well framed, the L* algorithm is nice and the guarantee accompanying it. Evaluation is done on 11 varied languages and the bridge between formal languages and neural languages is welcome.

### Weaknesses
The work is quite dense, and comes with a thorough appendix. I feel like it could almost have been more than a single paper. 

It would have been nice to see more variations in the trained language models (architectures, hyperparams) , as an empirical study of learnability, but again that could be another paper. And same for some public LLMs.

Comparison to other ways to identify the machines could maybe have been done. I am not sure what methods would make most sense but i imagine there are some.

Tiny nitpicks: it wasn't immediately clear to me from e.g. figure 2 on its own what was being plotted until after reading the work. Maybe say (time to run L*), (number of states in found machine). Also in figure 1, the labels for the symbols make sense but they could be somehow marked as such.

### Questions
Did you try at all to extract automata from public LLMs for simple languages to see how they implement something like parity? E.g. to benchmark them on whether they find minimal solutions or something of that sort. I see you point out this might be overly complex machines in the appendix, but I’m curious. 

Do you see a path forward to extending the work to WFSAs?

### Soundness
4

### Presentation
3

### Contribution
4
