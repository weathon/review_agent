# Provably Learning from Language Feedback

- Decision: Reject
- Scores: 2, 6, 2, 8

## Abstract
Interactively learning from observation and language feedback is an increasingly studied area driven by the emergence of large language model (LLM) agents. While impressive empirical demonstrations have been shown, so far a principled framing of these decision problems remains lacking.  In this paper, we formalize the Learning from Language Feedback (LLF) problem, assert sufficient assumptions to enable learning despite latent rewards, and introduce *transfer eluder dimension* as a measure to characterize the hardness of LLF problems.
We formalize the intuition that information in the feedback governs the learning complexity of LLF problems.  We demonstrate cases where learning from rich language feedback can be exponentially faster than learning from reward. We develop a no-regret algorithm, called `HeLiX`, that provably solves LLF problems through sequential interactions, with performance guarantees that scale with the transfer eluder dimension of the problem. Across several empirical domains, we show that `HeLiX` performs well even when repeatedly prompting LLMs does not work reliably. Our contributions mark an important step towards designing principled interactive learning algorithms from generic language feedback.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces a theoretical framework for Learning from Language Feedback (LLF), defining notions such as verifier, transfer eluder dimension, and a provable no-regret algorithm called HELiX. The authors also provide empirical tests and claim provable efficiency.

### Strengths
The paper attempts to provide a formal bridge between language-based feedback and provable learning theory, which is an important direction.

The introduction of a “transfer eluder dimension” to quantify information in feedback is conceptually interesting.

HELiX’s idea of hypothesis elimination and verifier-guided exploration is an appealing concept.

### Weaknesses
1. **Verifier Assumptions (Assumption 2) and Unbiased Feedback (Assumption 3)**

The central assumptions—**the Verifier assumption (Assumption 2)** and **the Unbiased Feedback assumption (Assumption 3)**—are **neither empirically testable nor adequately justified**.

The authors assume access to a verifier loss $\ell$ that perfectly reflects *semantic alignment* between hypotheses and feedback, but no practical method is given for constructing or validating such a verifier. While Appendix G argues that “LLMs are strong verifiers,” this is anecdotal; the authors should provide quantitative evidence (e.g., correlation between LLM-verifier loss and human-judged semantic consistency) to substantiate the claim that current LLMs satisfy the assumptions -- at least for the experiments that author did. 

Moreover, the claim that a verifier “can be implemented by prompting an LLM” is **untested**. There is no verification that the resulting LLM-verifier satisfies the required *bounded*, *eluder-finite*, or *unbiased* properties. At minimum, the authors should (i) describe how one could empirically estimate the **transfer eluder dimension** of an LLM-based verifier, and (ii) discuss whether this value is finite. If the transfer eluder dimension is effectively infinite, the regret guarantees become vacuous.

So I would recommend authors propose a constructive diagnostic for testing them (e.g., estimating transfer eluder dimension, or calibrating unbiasedness empirically).

---

2. **Lack of Theoretical Validation**

The main regret bound (**Theorem 1**) provides only an upper bound
$\tilde{O}(T^{3/4})$
under minimal assumptions on $\ell$.

* No **lower-bound analysis** is provided, making the claim that the rate is “optimal under $\ell_2$” might be misleading. Without citing or deriving lower-bound results for LLF or for transfer eluder dimension, it is unclear whether $\sqrt{T}$ is in fact optimal under $\ell_2$ assumption.
* The theoretical section seems like largely re-casts the standard eluder-dimension analysis in a new notation, though the introduction of the *transfer eluder dimension* is conceptually meaningful.

It would be accurate to report the $\tilde{O}(T^{3/4})$ bound as a conservative guarantee, but labeling it “optimal” under L₂ is not justified without matching lower bounds. I am fine with this is not a lower bound matching algorithm as it is meaning to provide a new concept. However, I think claiming lower bound on $\ell_2$ should be justified, or at least provide some references. 

---

3. **Disconnect Between Theory and Empirics**

The practical **HELiX (Algorithm 2)** deviates substantially from the theoretical Algorithm 1:

* The **verifier** is replaced by heuristic LLM-scoring functions rather than a loss $\ell$ satisfying Assumptions 2–3.
* **Hypothesis elimination** is implemented via “thought sampling + cross-verify” which lacks a formal connection to the theoretical framework.
* **No regret or transfer-eluder metrics** are reported—the figures show only cumulative reward curves (Fig. 2 & 6).

Consequently, the experiments **do not test the theoretical predictions**. It is unclear whether the improvements observed arise from the principles of Algorithm 1 (e.g., UCB-style exploration) or simply from LLM prompting heuristics

**Recommendation:** Report empirical **regret curves** relative to an oracle or to the best-in-hindsight policy, and evaluate whether empirical performance scales with proxy measures of the *transfer eluder dimension* (e.g., hypothesis diversity or verifier variance).

---

4. **Ambiguous Practicality and Computational Cost**

The paper lacks analysis of the **computational feasibility** of HELiX:

* There is no guidance on how the verifier loss $\ell$ or transfer eluder dimension could be computed or approximated in realistic LLM settings.
* Maintaining the hypothesis set is potentially exponential in |A|, yet the paper does not analyze resource or time complexity.
* The “thought-sampling” and “cross-verify” steps appear extremely compute-intensive—requiring quadratic LLM evaluations per iteration—but the paper omits runtime, model size, or token-budget statistics.

---

5. **Weak Experimental Evidence**

* The environments are **toy-scale** and synthetic (Wordle, Battleship, Minesweeper), providing limited insight into general LLF performance.
* There are **no statistical tests**, **no ablations** on verifier quality, hypothesis size N, or the influence of ref.
* Baselines (“No exploitation step”, “No $\pi_{ref}$”) are underspecified and not clearly motivated by theory.
* Reported metrics (cumulative reward) lack any linkage to the proposed theoretical quantities (regret, eluder dimension, unbiasedness).

Overall, the experiments are insufficient to substantiate the claimed theoretical contributions. The authors should include:

1. Ablations varying verifier fidelity and hypothesis diversity.
2. Explicit regret estimation against a known ground truth.
3. Runtime and scaling analysis.

### Questions
Mostly written in weaknesses section. 

My expertise lies between the theoretical and empirical aspects of this work, so there may be minor misunderstandings. I am open to revising my evaluation based on clarifications.

### Soundness
3

### Presentation
4

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper "Provably Learning from Language Feedback" proposes a theoretical analysis of in-context learning with LLMs for stateless RL tasks. To do so, authors introduce the transfer eluder dimension, which quantifies the hardness of the Learning from Language Feedback at hand. This theoretical quantity, based on the eluder dimension, allows to show that learning from feedbacks is at least as easy as from associated feedbacks, whenever the feedback is discriminative (i.e., is able to distinguish true hypotheses from false ones based on feedbacks to the same extent than based on rewards). Finally, the paper gives an illustrative algorithm that iteratively excludes hypotheses (and their associated actions) from the pool of candidates, in a similar way as UCB approaches, but based on language rewards rather than on observed rewards. Under some assumptions regarding the ability of a verifier agent to assess the consistency of the feebacks provided in response to a selected action for a given hypothesis, authors show that the regret of their algorithm is sublinear. Some experiments empirically complement this analysis.

### Strengths
- Important work to better understand in-context learning of LLMs in one step RL problems. This is a continuously growing field, with many works proposing to use LLM agents in a loop to achieve complex tasks, often greatly lacking of formalization. This theoretical work looks to propose a significant step to close that gap. 

- Theoretical assumptions and demonstrations look reasonnable and well sounded (I did not check all the proofs though)

- Authors made an important effort for illustrating abstract concepts with examples

### Weaknesses
-  Pedagogy and Positioning : While it is quite well written globally, I had a bit of difficulty fully grasping this paper. First, I believe the positioning should make it clearer that the work is set in a one-step RL (bandit) framework. It took me some time to realize that the paper does not consider the more common multi-step RL setting that I am more familiar with. I appreciate the authors’ effort to provide illustrative examples for the various abstract concepts — these are genuinely helpful. However, I think the discussions that justify the different formulations could be expanded and made more pedagogical. I found it rather challenging to interpret several elements of the paper. For instance, the conclusion of Example 1 could be more detailed to clarify the main takeaway. Similarly, the definitions are quite dense, and understanding their precise meaning requires effort. A stronger pedagogical focus in these sections would be highly appreciated. 

- Extension to the non deterministic setting: While authors claim their framework is directly applicable for stochastic environments, I feel it is not fully trivial

- Extension to the multi-turn return setting : Authors provide insights for the extension to a contextual setting (which should better further developped by the way). But nothing is said about the classical multi-turn return setting of the RL litterature (beyond bandits). Would it be possible to extend the work for cumulative discounted rewards / feedbacks ? This looks crucial for many complex tasks, which cannot explore efficiently if only taking utterances globally. 

- Experiments lack baselines and explanations. At least, I would have expected a classical UCB or Thompson Sampling baseline based rewards. Or even an epsilon-greedy exploration scheme (with or without a LLM as the agent). Many other agentic approaches could have also been experimented. For instance approaches based on TextGrad, which proposes backward steps in the form of a textual feedback.  Some ablations are not well described either. For instance, what $\pi_{ref}$ stands for ?  Experiments could have also been performed on more challenging environments, such as theorem proving for instance.

### Questions
1. Algorithm Helix assumes the availability of the optimal policy for any hypothesis from the pool. For simple settings where hypothesis are simply defined as an action (such as in the multiarm bandit setting), this is trivial. But I suspect that many other settings would hinder the applicability of the proposed algorithm. Please discuss. 

2. How would the approach compare to baselines mentionned in weaknesses ? 

3. How confidence levels can be set in Helix ? Is there a decreasing scheme as in epsilon greedy approaches ? Note that there exists a scheduling decrease that allow epsilon-greedy to be optimal (while unavaible in practical settings). 

4. Could author better precise what they mean by "rubric". I feel this is quite unclear, as examples given in discussion are quite complex, while what is considered for experiments is extremly simple. 

5. what $\pi_{ref}$ stands for in the abblations ?

6. How would perform the proposed algorithm in more complex environements (e.g., contextual / stochastic / implying reasonning) ? 

7. Would it be possible to extend the work to the multi turn RL setting (implying a sequence of interactions with an environment) ? 

8. Is the reward mapping availability a common practical assumption ? 

   


Minor remarks: 

- p7. "In addition to this example, one can check that... " ==> please mention that this can be found in appendix C4
- top of p7 : fist ==> first

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper studies how to do learning with language feedback and proposes a no-regret algorithm for the tasks. They review and formalize the LLF setting in section 3, defining the setup for the policy to maximize an unknown reward function given only token-based feedback from the actions. The emphasis here is that distinct from the usual RL/bandit setup, the policy only has information about the reward values through natural language feedback. From here, they also define a hypothesis space and verifier to complete the setup Then, S4.1 proposes to quantify information in the feedback by an extension of the eluder dimension to characterize how well the feedback reduces uncertainty about the unknown reward function, and the rest of that section builds up to their UCB-style algorithm HELiX, furnishing a regret bound in Thm 1. Experimentally they evaluate on wordle, battleship, and minesweeper.

### Strengths
Language feedback is a rich additional source of information and under-explored topic, as the language models and tasks have only recently enabled progress in this area. In a research landscape where many language tasks and methods are ill-defined and not possible to make theoretical statements about, I am generally supportive of more formalization in the LLF direction so more precise mathematical statements can be made about improvement and learning in language spaces. The paper has a nice arc building up to HELiX and the resulting regret bound, and the experiments in Figure 2 demonstrate how HELiX is better than a greedy baseline on the tasks.

### Weaknesses
Despite being generally positive about formalizing this space, the experimental results are holding me back from giving an acceptance at this point. I am very open to discussing this.

Before getting into the specific formal setting of the paper, Wordle, Battleship, and Minesweeper seem like well-studied tasks where other methods and experimental settings have been investigated. However, the results in Figure 2 are isolated from these, and not compared to what I would consider state-of-the-art AI methods on these settings. Reporting the ablations across cumulative reward is interesting for comparing bandit-like methods, but seems slightly disconnected from actually knowing how well the performance on these settings are. This makes it difficult to understand how significant of an experimental advance the paper provides and leads me to an interpretation that the main contribution of the paper is more in the theoretical contribution instead of being experimentally SOTA. Slightly beyond this, why only select these three tasks, and not evaluate on all of the tasks from LLF-Bench?

This could be okay, but the domain of the paper is an experimental topic, and "provable learning from LF" is a strong title. On this, "provable" still needs some scoping, e.g., only to the setting they formalized.

And lastly, there are many other pieces of related work that use language-based feedback, such as many of the coding agents (SWEAgent, OpenHands, and the citation graph around these), as well as DSPy and TextGrad. All of these settings also iterative improve a textual object based on text-based feedback in many cases. They are not mentioned in the submitted paper. I believe they would minimally be interesting to conceptually connect to. They could be interpreted as using language feedback "without" provable guarantees, so it would be very insightful for these communities to understand how LLF with provable guarantees can help improve them. And it would be an extremely strong paper to compare exactly to some of their exact and previously-published experimental settings. My guess is that it may be difficult to get it exactly in the setting considered in the paper, but maybe they could be minimally be considered as baselines on the tasks in this paper?

### Questions
I am very open to discussing my experimental concerns and connections to agents/DSPy/TextGrad throughout the rest of the review period

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper studies a model of online learning with rich feedback called "learning from language feedback" (LLF), motivated by online decision making applications where the decision maker can receive language-based feedback from e.g., LLMs. In the proposed model, there is some ground truth hypothesis \eta^* unknown and is known to be in class H; and the learning agent repeatedly takes an action a_t and receives an observation o_t from the observation distribution f_{\eta^*}(a_t). Its goal is the maximize its cumulative reward \sum_{t=1}^T r_{\eta^*}(a_t), where the mapping r_{\eta}(a) is known ahead of time. The paper proposes a new concept called the ``transfer eluder dimension'' that measures the statistical complexity of the LLF problem. It further shows that:
- (Section 4.3) under some assumptions, LLF is no harder than learning from reward feedback (structured bandits) 
- (Section 4.2) in some settings, LLF can be exponentially easier than learning from reward feedback

 It also proposes the HeLiX algorithm, which enjoys a sublinear regret, given that the transfer eluded dimension is finite. Experiments show that HeLiX can outperform baselines such as greedy.

### Strengths
- The paper provides a nice conceptual framework to understand the benefit of learning from language feedback, which is relevant in modern applications where LLM are good at providing language feedback beyond rewards

- A general statistical measure, transfer eluder dimension is proposed, that can handle general hypothesis classes. 

- The initial set of theoretical results are somewhat complete, in the sense that it compares with reward-based feedback, as well as showing when it can be significantly more useful

### Weaknesses
- Assumption 3 seems important, although maybe necessary for clear development of theoretical results. Do you have a sense if this is satisfied in your experiments?

- (Clarity) Can the authors clarify what are some roadblocks in proving \sqrt{T} regret bound for HELiX under the general feedback setting (beyond the square loss assumption)?

- I agree with the final remark by the authors that there are some LLF problems with infinite transfer Eluder dimension and are trivially solvable. I wonder if we can show something like a converse of Theorem 1, e.g. for every d, there exists a class of hypotheses, observation mapping, and reward mapping with transfer Eluder dimension <= d, such that the regret of any algorithm is at least Omega( \sqrt{d} T^{3/4} ). A related question is, is O(T^{3/4}) regret bound in Theorem 1 fundamental to some loss \ell?

### Questions
- In Sec. 4.4 and Alg. 1, the \pi's are meant to be action a's?

- Without the consensus exploitation step, can HELiX still have a regret bound as in Theorem 1? That step seems rather nonstandard to me. In other words, why don't we just use a pure optimism-based algorithm?

### Soundness
3

### Presentation
4

### Contribution
3
