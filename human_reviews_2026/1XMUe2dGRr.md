# Towards Persistent Noise-Tolerant Active Learning of Regular Languages with Class Query

- Avg Score: 4.80
- Decision: Accept (Poster)
- Scores: 2, 6, 6, 2, 8

## Abstract
Large Language Models (LLMs) are increasingly deployed in human–AI collaborative decision-making systems, where they are expected to align precise formal representations with ambiguous natural language. However, their ad hoc strategies for resolving ambiguity often lead to hallucinations and inconsistencies. We formalize this setting via probabilistic Minimally Adequate Teachers (pMATs) that (i) answer membership queries with fixed but possibly flipped labels, and (ii) return valid counterexamples to hypothesis equivalence. We present **CAPAL** (**C**lass-query **A**ctive, **P**ersistent-noise-**A**ware **L**earning), an active algorithm for learning deterministic finite automata (DFAs) that remains correct under persistent membership noise without demonstrations. CAPAL augments the classic \$L^\star\$ loop with two components grounded in our implementation: (1) a *class query* realized as a statistical same-state test that compares disagreements between two prefixes against a noise-floor estimate \$\hat{\eta}\$ with Hoeffding tolerances; (2) a *discrimination tree* that selects a near-minimal discriminator, keeping the core suffix set small. An efficient micro-bootstrap and cache-reuse scheme estimates \$\hat{\eta}\$ with few new queries. We prove convergence given a perfect language-equivalence oracle and show substantial membership-query savings in practice. Our evaluation across multiple benchmarks, including RegexLib and KB13, demonstrates that this approach enhances both the efficiency and robustness of DFA learning under noisy oracles, supporting the view of LLMs as fallible yet useful collaborators for synthesizing verifiable formal artifacts.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper studies a new learning theory problem for regular languages, have a noisy membership query (MQ) oracle, but a perfect equivalence query oracle in a L* learning setup. Such noisy MQ oracles could be LLMs, though I am not sure why they are needed for the simple problems studied. The proposed algorithm CAPAL shows high accuracy with fewer calls to the MQ oracle.

### Strengths
1. I like the problem formulation of pMAT and the recognition of persistent errors.
2. The work studies a variety of prompting strategies and compares them for their effectiveness within membership query oracles. 
3. The code-based oracle is pretty interesting and gives consistently better results.

### Weaknesses
1. My biggest criticism of the paper is in the motivation. I think the abstract and intro could provide more evidence for the utility of the problem studied from the practical perspective, with detailed, concrete examples, especially involving LLMs for membership queries. The abstract doesn't introduce the problem and is hard to follow.
2. I question whether the assumption of a perfect EQ oracle is realistic. For the experiments, how is the perfect EQ oracle implemented? How would we get a perfect EQ oracle for practical problems?
3. The main algorithm/contribution, CAPAL's description is pretty dense and hard to follow. Multiple crucial steps are introduced, but there is less background provided to fully understand and appreciate it.

I recommend rejection mainly for the paper's writing, which currently lacks motivation for the problem and its assumptions and is pretty dense. I encourage the authors to simplify the writing, maybe by including some examples, which can make their contributions more accessible.

### Questions
1. Why is the code based oracle deterministic? In repeated samples from the MQ oracle LLM, we could get different codes and hence even this scenario is stochastic?
2. I wonder whether the CAPAL algorithm necessarily needs LLMs. What would it look like theoretically and empirically for any other kind of  noisy MQ oracle, other than LLM?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper tackles the challenge of using Large Language Models (LLMs) to learn formal representations, specifically deterministic finite automata (DFAs). 

It proposes a new learning model called the Probabilistic Minimally Adequate Teacher (pMAT). This framework assumes the LLM acts as a noisy membership query (MQ) oracle but is paired with a perfect equivalence query (EQ) oracle that provides definitive counterexamples to incorrect hypotheses.

The paper presents CAPAL (Class-query Active, Persistent-noise-Aware Learning), an algorithm designed to learn Deterministic Finite Automata (DFAs) correctly within the pMAT model. Instead of trusting individual noisy answers, CAPAL uses a statistical class query to determine if two prefixes belong to the same state and a discrimination tree to efficiently refine its hypothesis with minimal new queries.

Experiments show that CAPAL significantly outperforms standard active learning algorithms in noisy environments. The study also finds that having the LLM generate a code-based oracle (a deterministic program to answer queries) is far more effective than direct prompting, reducing errors by over 90% and cutting LLM calls to just one per task.

### Strengths
1.  It introduces a novel learning framework (pMAT) and a provably correct algorithm (CAPAL) to solve it. The paper formalizes the problem of "persistent noise," where an LLM is consistently wrong about certain queries, which is a more challenging and practical scenario than random errors. The proposed CAPAL algorithm is specifically designed to be robust to this noise by using a statistical "class query" and is backed by a formal proof of convergence, demonstrating its theoretical soundness.
2.  It provides a solution with the "code-based oracle." The paper's practical finding is that using an LLM a single time to synthesize a deterministic program (a code-based oracle) is far more effective than using it for repeated queries. 
3. The claims are supported by a thorough and rigorous experimental evaluation. The authors validate it extensively against multiple different active and passive learning baselines across multiple benchmarks.

### Weaknesses
1. The entire theoretical guarantee and the practical success of CAPAL hinge on the assumption of a perfect, noise-free Equivalence Query (EQ) oracle that always returns a valid counterexample. This is a limitation, as real-world verifiers (whether human or automated) are often fallible. The paper acknowledges this but does not investigate the consequences. The work could be strengthened by including experiments with an imperfect EQ oracle (e.g., one that fails to find a counterexample with some probability). 
2. The algorithm's strategy for a "label-only" counterexample (where a state is simply mislabeled as accepting/rejecting) is to wait for the exact same counterexample to reappear before escalating by adding its suffixes as discriminators . This seems potentially inefficient. A different counterexample that reaches the same mislabeled state would not trigger this escalation, possibly leading to many wasted EQ cycles to fix one incorrect label. The paper would be more convincing if it justified this specific design choice with evidence.

### Questions
1. The paper's entire theoretical and practical framework relies on a perfect, noise-free $O_{EQ}$. How do you expect CAPAL to perform if this oracle is imperfect? For instance, what happens if the $O_{EQ}$ fails to return a counterexample for an incorrect hypothesis (a "false negative") or provides a spurious counterexample for a correct one (a "false positive")?
2. The paper states that for label-only counterexamples (where the DFA's structure is correct but a state's acceptance is wrong), the algorithm only escalates by adding all suffixes of c to $E_{core}$ if the exact same counterexample c reappears. What is the specific rationale for this "repeat-only" escalation policy?

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
The paper introduces a probabilistic framework for learning formal languages from large language models (LLMs) that act as noisy oracles. The authors formalize this setting through probabilistic Minimally Adequate Teachers (pMATs), which can provide incorrect answers to membership queries but always return valid counterexamples for failed equivalence queries. Within this framework, they propose CAPAL (Class-query Active, Persistent-noise-Aware Learning), an algorithm for actively learning deterministic finite automata (DFAs) that remains theoretically correct even when the oracle produces persistent labeling errors. CAPAL extends Angluin’s classic L* algorithm with statistical same-state tests that distinguish language classes under bounded noise, and with a discrimination tree intended to keep the hypothesis compact. The method also estimates the oracle’s noise rate using a lightweight bootstrap procedure that minimizes redundant queries. The authors prove convergence of CAPAL under a perfect equivalence oracle and show empirically that in certain settings the method requires fewer membership queries than existing approaches. More specifically, when applied to datasets such as RegexLib and KB13 the method shows improved robustness in DFA learning from noisy oracles. Overall, the results support the idea that LLMs, despite their fallibility, can serve as useful collaborators for synthesizing verifiable formal models from natural language.

### Strengths
The idea of using LLMs to guide automata learning algorithms is interesting. Their implementation CAPAL provide evidence that  active learning of regular languages can be made robust to persistent membership noise when a perfect equivalence oracle is available. The main approach is to shift from trusting individual MQ labels to issuing a class query that aggregates evidence against a principled noise floor, with early stopping and monotone (negative) caching to avoid re-litigating pairs already shown different.

### Weaknesses
The main problem is that the approach does not work if the equivalence oracle is approximate, and no extension to the approximate case is provided . This is an issue because perfect equivalence oracles are typically expensive, and in practice approximate equivalence oracles are used.

### Questions
No question.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper studies active learning of regular languages with membership queries and equivalence queries.This problem regained recent interest due to the usage of LLMs in converting high-level description of regular languages to a deterministic automata, where LLM can serve as an oracle in answering these two types of queries. The paper assumes that responses to membership queries can have persistent noise, while the response to equivalence queries are noiseless. The paper proposes CAPAL algorithm and establishes its correctness guarantees. Experiments show that the proposed algorithm has lower number of queries and a better learning accuracy.

### Strengths
- I think the problem motivation is interesting. Converting an informal language description to a formal language allows constructing guardrails of agents, ensuring agent's safety and avoiding unintended consequences in a provable way. 

- The experimental results look good to me

### Weaknesses
- I am not sure if assuming the equivalence query oracle is noiseless is practical. Such oracle is also implemented using LLM in practice, am I understanding correctly? Then they may hallucinate?

- I wasn't able to follow the presentation of the CAPAL algorithm, perhaps due to me not being an expert in active learning with regular languages. For example, in line 6, how is sa ~ u defined - is it an equivalence relationship defined by the partition in line 5? In line 8, what are the nodes of the decision tree, and what are the splitting criteria for the tree nodes? 

Also, it is mentioned in line 307-308 that the membership answers equals to the true label with probability 1-eta with eta < 1/2 -- is this for all the queries? If so, then this seems to be inconsistent with Definition 3.2 (where it is mentioned that for some query, the mode of the oracle's response distribution disagrees with the truth)? 

Also, is the label flipping probability assumed to be homogeneous across all queries in theoretical analysis? It seems not the case from the empirical result in Figure 2. 

- The paper provides some correctness guarantees of the algorithm (Theorem 1 and 2). It relies on some assumption on the knowledge of the safe noise cap \bar{\eta}. Can this be known, or estimated? Alg. 1 line 2 proposes to estimate \hat{\eta}, but not enough details are given. 

- Is there some formal theorem query complexity that can be given? Without that the theory seems incomplete.

### Questions
See questions above.  Also, can the authors comment on why in Figure 3, CAPAL's success rate is not always 1 for all epsilon?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This work considers the problem of learning a deterministic finite automaton from a natural language derived oracle. Specifically, an LLM as a membership oracle to answer yes/no questions about a natural language description of a formal language. This can either be done by answering questions one at a time or converting the description into code, e.g., python. A fundamental problem is that the LLM may make systematic errors in labeling queries. This can result in learning arbitrarily incorrect languages. This is classically addressed by the LEARNANYWAY algorithm, however requires small counterexamples to be made practical. This however is typically not the case as counterexamples are often found via formal analysis or random search/conformance testing.

The contribution of this work is CAPAL algorithm which seems to perform fairly efficiently even in the presence of labeling errors. The key assumption, called pMAT, is that in addition to a natural language membership oracle (with the possibly to err) there is a perfect counter example oracle. The algorithm is relies on a combination of statistical tests and clever usage of a discrimiation tree, ala Kearns and Vazirani.

### Strengths
The idea of extracting formal structure from natural language is timely and important. While easy to generate such things in an ad-hoc manner, e.g., code gen, this and related work help provide rigor and guarantees to such procedures. The focus on DFAs is interesting as it is the simplest structure that allows for memory, being the target of decades of learning research.

The techniques are as far as I know novel, and the framing of pMAT makes the underlying assumptions and limitations fairly clear. While clearly strong, see weaknesses, it allows for progress made in understanding this problem.

### Weaknesses
1. The primary weakness of this approach is the very strong assumption of an equivalence oracle. While there are a few settings this might make sense, e.g., when an (expensive) verification process exists and is cheap compared to the LLM query required for program translation, it obviously not typically available. In such settings conformance testing and random sampling are much more common, typically yielding PAC like guarantees. This seems like a natural next step, and is explicitly acknowledged in the discussion.

2. Another minor writing point is that, in exposition, this paper leans heavily on existing knowledge of automata learning. While not strictly required, I think without a fairly good automata learning background, a reader will likely bounce off of this work. This is particularly likely with much ICLR community. Nevertheless, I think the paper is understandable.

3. Finally, a quick nit pick about exposition with the L*LM related work. My understanding is that the demonstrations only come into play as a mechanism for refining the language *after* an automata has been guessed using a classic learning algorithm. One should able to drop in CAPAL directly into the L*LM framework and still leverage the demonstration modality right?

### Questions
See weakness #3.

Also, do you have a sense how this would compare against SAT based approaches allowing for up to `k` labels being flipped (where `k` is derived from the noise floor?) Presumably, this is more scalable, but on the hand works like L*LM showed that creating distinguishing sequences of the smallest consistent DFAs resulted in less hallucination opportunities.

### Soundness
4

### Presentation
3

### Contribution
3
