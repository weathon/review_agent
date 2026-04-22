# Taming Imperfect Process Verifiers: A Sampling Perspective on Backtracking

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 8, 8, 4, 6

## Abstract
Test-time algorithms that combine the *generative* power of language models with *process verifiers* that assess the quality of partial generations offer a promising lever for eliciting new reasoning capabilities, but the algorithmic design space and computational scaling properties of such approaches are still opaque, and their benefits are far from apparent when one accounts for the cost of learning a high-quality verifier. Our starting point is the observation that seemingly benign errors in a learned verifier can lead to catastrophic failures for standard decoding techniques due to *error amplification* during the course of generation. We then ask: can this be improved with more sophisticated decoding strategies?

We introduce a new process-guided test-time sampling algorithm, VGB, which uses theoretically grounded *backtracking* to achieve *provably* better robustness to verifier errors. VGB interprets autoregressive generation as a random walk on a tree of partial completions, with transition probabilities guided by the process verifier and base model; crucially, backtracking occurs probabilistically. This process generalizes the seminal *Sinclair-Jerrum random walk* (Sinclair & Jerrum, 1989) from the literature on approximate counting and sampling in theoretical computer science, and a conceptual contribution of our work is to highlight parallels with this literature. Empirically, we demonstrate on both synthetic and real language modeling tasks that VGB outperforms baselines on a variety of metrics.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
In test-time scaling, process verifiers guide generation towards high-reward outputs. However, process verifiers may be inaccurate. This paper shows that process verification errors can "cascade" in the sense of growing with the number of decoding steps. It then presents an alternative test-time scaling algorithm, VGB, which is shown to have positive theoretical properties with regard to multiplicative errors in the process reward: 

- Given a uniform bound on the multiplicative process reward error, VGB is guaranteed to sample from a distribution $\hat{\pi}$ with bounded total variation distance from the reward-optimal sampling distribution $\pi^*$, and to output a "leaf node" (a complete generation) in time $\tilde{O}(H^3)$.
- Given a bound on the expected ratio of the estimated and true process rewards, the output distribution is similar to the reward-optimal sampling distribution with high probability. Specifically the ratio $\pi*/\hat{\pi} < \delta^{-1} 48H(1+\epsilon_V)^2$, with $H$ indicating the decoding length and $(1+\epsilon_V)$ the multiplicative bound on the expected ratio of estimated and true process rewards. The runtime in this setting grows as $\tilde{O}(H^5)$.

The paper makes several other theoretical contributions, including showing that other error conditions are insufficient to achieve the desired guarantees (appendix D).

The empirical side focuses mainly on minimal synthetic language tasks, such as Dyck grammars.

### Strengths
The paper addresses an important practical issue in test-time scaling. It introduces useful theoretical machinery, which to my knowledge is novel in this domain. The discussion of "pitfalls" of existing techniques is clear and convincing, as is the presentation of the theoretical guarantees offered by the approach.

The empirical results shown in the main paper mostly support the main claims, albeit in toy settings: fig 1 demonstrates that VGB dominates block best-of-N and block rejection sampling in the Dyck grammar task, and fig 2 shows that the KL-divergence error grows more slowly with $H$ in the ABC task.

Overall, while I doubt that the specific algorithm here is likely to be of practical value, the paper seems like a significant theoretical advance that will catalyze more research that may ultimately make test-time scaling more effective and/or efficient.

The writing quality is excellent throughout the paper.

### Weaknesses
While the paper claims to address the problem of cascading errors, it seems like the bound in theorem 4.2 still grows in $H$. Similarly, the KL-divergence errors seem to grow with $H$ in Figure 2, although they do grow more slowly than with action reject sampling. Thus it seems more like the proposed approach *mitigates* this issue rather than solving it.

I think the proposed algorithm is unlikely to be adopted due to the polynomial run-time scaling with $H$ (cubic with the uniform guarantee, and quintic with the expectation condition). But as noted above, that's not a deal-breaker to me because the paper may stimulate research that results in more efficient algorithms. The authors do acknowledge this issue in the paper.

I am not a specialist in this area, and I found it pretty hard to understand how the algorithm actually works. In particular, the paper did a poor job explaining the search space explored by the backtracking algorithm, and would have benefited from a better figure than fig 1 (right).

### Questions
Please feel free to comment on the points raised above: (1) whether errors still increase with $H$ in VDB, and (2) whether and how the algorithm can be practical given the apparently poor polynomial scaling with $H$.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper proposes a generalization of the Sinclair-Jerrum random walk to reward-guided generation of language models when the value function is not perfect.
The proposed method (VGB) has a strong theoretical motivation, and where possible, formula derivations are delegated to relevant work.

### Strengths
+ Solid theoretical foundation backed up by experimental results
+ Simple examples that illustrate the problem of a more greedy approach (ActionLevelRS)
+ Thought out collection of evaluation tasks (such as parity that is not in $\mathrm{AC}^0$)
+ Polynomial (in generation length, action space, value function error, and target $D_{\mathrm{TV}}$) algorithm for sampling from an optimal policy for a KL regularized objective

### Weaknesses
+ Minor: Relatively simple generation tasks (Dyck grammar, Python test generation, and letter avoidance with output length 32 and  0.5B parameter model). To be more precise, inclusion of simple tasks benefits the work; however, inclusion of more complex tasks should further help to assess the practicality of the method.
+ Minor: The paper delegates a lot of proofs to the Appendix. To be more precise, the Appendix consists of around 55 pages, and around 25 of them are proofs. This is a highly dense, in particular for a conference where "Authors may use as many pages of appendices (after the bibliography) as they wish, but reviewers are not required to read the appendix.". I must say I haven't managed to check all the proofs of the propositions and theorems. However, I noted that some of them seem to miss some (relatively obvious so this is a minor problem) assumptions or have some typos regarding the parentheses (see proof of proposition G.3).
+ Minor: the algorithm requires a lot of sampling to achieve theoretical guarantees

### Questions
Do authors have any intuition or results regarding more complex problems, especially those that have a value function that is much harder to learn/approximate?
To be more precise, the algorithm still requires a good, on average, value function for efficient sampling, and it is not clear whether, in more complex cases, it wouldn't be cheaper to tune the policy with GRPO instead.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates the robustness of process-guided test-time alignment in generative language models, focusing on settings where imperfect value functions (verifiers) are used to steer generation. The authors identify error amplification pitfalls associated with standard action-level sampling when value-function estimates are used, and propose a new algorithm, Value-Guided Backtracking (VGB), grounded in Markov Chain Monte Carlo theory to counteract these weaknesses. They provide theoretical analyses and guarantees for VGB—even under average-case error regimes—as well as empirical validation across synthetic, code generation, and constrained text generation benchmarks.

### Strengths
- The authors provide an explicit theoretical understanding of how small, seemingly benign errors in learned process verifiers can catastrophically degrade the distributional fidelity of generation under standard action-level sampling. 
- VGB is a principled modification to existing sampling strategies, and the authors convincingly root its correctness and performance in established Markov chain (Sinclair-Jerrum walk) theory.
- The paper benchmarks VGB against both sampling (Block Best-of-N, Block Rejection Sampling) and search-style (beam search, locally constrained decoding) baselines across multiple axes.

### Weaknesses
- While the theoretical grounding is new, its real-world applicability and improvement magnitude over clever heuristics or simple extensions (e.g., adding "undo" moves) is not novel, especially since the best practical variant of VGB may require hyperparameter tuning for the backtracking probability, which the authors acknowledge but do not fully explore.
- The computational cost of VGB, especially for long horizons (scaling as $\tilde{O}(H^3)$ for guaranteed leaf sampling under uniform error, and worse under average-case errors), could render it prohibitive for many realistic LLM tasks beyond toy or controlled environments.
- Most evaluation focuses on (relatively) small-scale or synthetic tasks—such as the Dyck grammar and ABC tasks. Despite one code generation task, there are no results on multi-turn question answering, math and code, or agentic reasoning where process verifiers are essential and imperfect. This leaves open how VGB would fare in "hard" settings.

### Questions
- Can the authors rigorously clarify the scalability of VGB for long-horizon, real-world LLM settings, both in terms of query count and wall-clock time, beyond what is presented in synthetic or small-scale experiments? For example, can VGB be realistically run for open-ended generation and complex code synthesis tasks at scale, or do computational bottlenecks dominate in these regimes?
- Would the main conclusions regarding error amplification and backtracking hold if the value function $\widehat{V}$ were implemented with more complex architectures (e.g., transformer-based verifiers) or noisy/biased sampling of partial completions? Are the present synthetic experiments sufficiently representative?
- Is there a principled way to tune the backtracking parameter (probability or weight), especially in the absence of oracle rewards or in scenarios where the true error profile of $\widehat{V}$ is unknown?
- Do the authors have insight into failure modes for VGB in more diverse or adversarial settings (e.g., prompts specifically constructed to trip up process verifiers), and could VGB be made robust to such cases?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors study "process verifiers" for LLMs within a theoretical framework. The notion of value function is introduced. It is supposed to align LLM generation with a predetermined reward. The "true" value function is defined, as well as its approximate counterpart. The central question of the paper is: how can we sample from the LLM using signal from the approximate verifier so that the resulting distribution of responses remain close to the optimal one achieved with the "true" value function?

A VGB sampling algorithm based on Sinclair-Jerrum random walk is proposed and its theoretical guarantees are proved under certain assumptions.

Additionally, some small-scale experiments are conducted showing the practical usefulness of the method.

### Strengths
* I find the topic of the paper interesting and timely.
* The authors put effort into introducing a formal framework of sampling from LLMs with verifiers and seem to prove significant theoretical results.
* The presented VGB algorithm is conceptually simple.

### Weaknesses
1. My main objection is that it's really unclear to what extent the presented theoretical results are relevant in practice. The authors motivate the paper practically, mentioning LLM-based PRMs from recent literature. However, I'm not sure if the introduced theoretical framework captures these practical PRMs and their usage. More specifically, I'm not sure if Assumptions 4.1 or 4.2 will be satisfied in practice with real-life verifiers. 
2. The presentation in some places is not polished. Some symbols are not defined (for instance delta in l. 64, A^H in l. 65, D_TV and omega in l. 209), which sometimes makes reading / understanding more difficult than necessary. 
3. It is good that the authors include some number of experiments, but they are small-scale and targeting simple problems. I think they are not convincing of practical relevance of VGB and its theoretical properties.

### Questions
1. In your framework you fix the length of generation to be H. Can the framework be reformulated to assume variable length generation? 
2. In VGB the probability of staying in the current node is 1/2. Is this the only constant preserving the properties of VGB, or it can be different?

### Soundness
3

### Presentation
2

### Contribution
3
