# Learning Provably Correct Distributed Protocols Without Human Knowledge

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 6, 4, 2

## Abstract
Provably correct distributed protocols, which are a critical component of modern distributed systems, are highly challenging to design, and have often required decades of human effort. These protocols allow multiple agents to coordinate to come to a common agreement in an environment with uncertainty and failures, e.g., agents only observe their own state, messages can be lost, and agents may crash. We formulate protocol design as a search problem over strategies in a game with imperfect information, and the desired correctness conditions are specified in satisfiability modulo theories. However, standard methods for solving multi-agent games fail to learn correct protocols in this setting, even with then number of agents is small. We develop a learning process we call GGMS that combines a specialized version of Monte Carlo Tree Search with a transformer action encoder, a global depth-first search to break out of local minima, and repeated feedback from a model checker. We show that, under mild assumptions, this process will not miss correct solutions. In experimental validation, we show that GGMS can learn correct protocols for larger settings than existing methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces GGMS (Guided Global Monte Carlo Search), a framework for automatically synthesizing provably correct distributed protocols without relying on human-designed examples. The approach integrates Monte Carlo Tree Search (MCTS), Depth-First Search (DFS), and model checking, iteratively exploring protocol candidates and verifying their correctness through formal verification tools. The authors demonstrate GGMS on small-scale atomic commit and consensus tasks, showing that it finds correct protocols more reliably than baseline approaches such as standard MCTS or MCTS+DFS.

I should note that I am not an expert in distributed systems or formal verification, and my evaluation is therefore based on a non-specialist perspective. My review focuses on clarity, accessibility, and the apparent coherence and novelty of the approach, rather than on detailed technical verification.

### Strengths
- **Ambitious and conceptually strong idea.** Automating the synthesis of provably correct distributed protocols is a highly ambitious and forward-looking goal. While foundational protocols such as Paxos already exist and are widely implemented, the authors target a different problem: automating the design and verification of such protocols. In practice, even small variations in assumptions (e.g., timing, fault tolerance, message loss) require new, carefully reasoned protocols. Automating this process could reduce the time and human expertise needed to design correct distributed systems, which justifies the research direction despite the existence of established solutions like Paxos.

- **Clear motivation.** The introduction effectively conveys why distributed protocol design is difficult and how formal verification guarantees robustness. The analogy to AlphaGo-style learning is intuitive, though distributed protocols require perfect correctness rather than probabilistic success.

- **Methodologically coherent.** The integration of MCTS, DFS, and model checking appears consistent and well justified. The paper’s internal logic is clear, and limitations are discussed transparently.

- **Honest discussion of limitations.** The authors openly acknowledge the synchronous-network assumption, limited scale, and prototype nature of their implementation.

- **Potentially broad implications.** If extended successfully, this line of work could influence future research on verified ML and the automation of safe system design.

### Weaknesses
- **Accessibility.** Section 3 (Modeling a Distributed Protocol with Zero Knowledge) is difficult to follow for non-experts. The abundance of formal states and transitions interrupts readability. A more intuitive overview before the formalism would help.

- **Figures.** Figures 1–4 are hard to read due to small font sizes and dense labeling. Redrawing them with larger fonts and clearer legends would improve accessibility.

- **Empirical scope.** Evaluation is restricted to small-scale examples (atomic commit, consensus). It remains unclear whether GGMS could handle larger or asynchronous settings.

- **Nature of the contribution.** It is somewhat unclear whether the paper’s emphasis is theoretical or empirical (demonstration on simple tasks). The small-scale experiments and my lack of experise make it hard to assess which aspect dominates.

- **Background and related work overlap.** The long background and short related-work sections could be merged for better flow and a unified perspective.

- **Future work section.** Listing implementation upgrades (rewriting in C, GPU acceleration) feels too engineering-focused. A more research-oriented outlook—e.g., scaling to asynchronous or Byzantine models—would be stronger.

### Questions
- Could the authors clarify where exactly the claim “freezing an entire MCTS path does not cause missing of correct protocols” is proven or justified? Is this statement formally captured, or should the reader understand it as an empirical observation?

- On page 7, the paper mentions difficulties in “propagating the effects of DFS freezes through MCTS” and refers to the need for sufficient “propagation power.” Could the authors please clarify what is meant by propagation power in this context. Additionally, would the use of set-invariant or permutation-invariant architectures (e.g., Set Transformers) potentially help enforce more consistent transitions across related states?

- Does GGMS rediscover existing protocols such as 2PC or produce novel but functionally equivalent variants? Would it be expected of GGMS to find exisiting protocols?

- How does the algorithm scale in general? 

- All experiments are run on CPUs using a single-threaded Python implementation. Could the authors clarify whether this choice reflects a limitation of the GGMS algorithm (e.g., due to sequential dependencies in MCTS and model checking), or if it was simply an implementation decision? Do you expect the method to benefit from GPU acceleration or distributed parallelization in future versions?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents GGMS, a learning framework for automatically designing provably correct distributed protocols without human knowledge. The authors formulate distributed protocol design as a search problem in a game with imperfect information, where processes must coordinate to reach consensus despite partial observability, message losses, and potential crashes. GGMS combines a specialized Monte Carlo Tree Search variant inspired by Alpha-Zero with a transformer-based action encoder, global depth-first search to escape local minima, and iterative feedback from a model checker to guarantee correctness properties specified in SMT. Authors include experimental validation on synchronous atomic commit and consensus protocols.

### Strengths
I believe the authors tackle a challenging problem, automatically learning provably correct distributed protocols without human knowledge. I found quite intuitive the formulation of protocol design as a search problem in an imperfect information game. However I am not familiar with these protocols and I cannot comment on whether this is the first work attempting that. Nevertheless, The GGMS framework introduces several innovations that go beyond traditional MCTS including the integration of global depth-first search (DFS) to escape local minima while maintaining theoretical guarantees of eventual convergence. The authors also introduce a "Guided MCTS" component, which uses counterexamples from the model checker to bias the search toward problematic scenarios. 

In general the paper gives a sound theoretical foundation given the overall assumptions and the experimental results show the gains with respect vanilla MCTS

### Weaknesses
The paper makes several restrictive assumptions that significantly limit its applicability to real-world distributed protocols -although clearly highlighted by the authors-: 1) The method is restricted to synchronous networks 2) GGMS assumes that processes can only send identical messages to all other processes (rather than different messages to different recipients) 3) The usage of model-checking, which is computationally exhaustive, restricts the number of processes that can be checked.  The experimental results confirm severe scalability limitations - the largest evaluated setting involves only 4 processes with 2 failures. The exponential growth in running time (Figure 4) and the dominance of brute-force model checking in the verification phase suggest the approach will struggle with realistic protocol sizes. The authors mention switching to Z3 for verification but provide no evidence this would resolve the scalability bottleneck. 

The paper's introduction assumes significant background knowledge in distributed systems, making it harder to a general ML audience like ICLR. The first two pages dive directly into technical concepts like "atomic commit," "consensus," and "Byzantine Fault Tolerance" without providing intuitive explanations or toy examples. A simple running example (e.g., 3 processes trying to agree on a value with 1 possible failure) introduced early would greatly improve accessibility. The formal problem definition in Section 2 jumps into notation-heavy descriptions without first providing intuition about what a distributed protocol actually does.

I also found the empirical evaluation somewhat limited with it being  restricted to only two protocol types (atomic commit and consensus) in their simplest forms. I also missed ablations on the different modifications to MCTS that authors introduced.

### Questions
I don't have particular questions but I want to conclude that despite the restrictive assumptions, these don't hinder the contributions. I see this more as a grounding work and I don't see these assumptions as reason for rejection. I do think that including running examples on the introduction and ablations would improve the clarity and soundness of the paper.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes GGMS, a framework to learn distributed protocols that are provably correct, using only the desired properties and no human protocol knowledge. The approach models each process as a deterministic state machine and uses: (1) MCTS guided by a transformer policy, (2) a global DFS that “freezes” ambiguous transitions to break out of local minima, and (3) repeated feedback from a brute-force model checker that provides counterexamples for further training. The authors argue GGMS will not miss correct solutions under mild assumptions and show experiments on synchronous atomic commit and consensus, where GGMS achieves higher success rates than MCTS or MCTS+DFS within time limits, or small numbers of rounds.

### Strengths
1. The paper introduces a new integration of guided MCTS, DFS freezing, and model-checking feedback for protocol synthesis. It provides theoretical results ensuring that freezing does not exclude correct solutions.

2. The clarity is good. Assumptions are stated (synchronous, same state machine per process, last-round decisions). Pseudo-code and examples make the method understandable. The modeling of processes as deterministic state machines, the learning–verification loop, and counterexample-guided retraining are clearly explained.

3. Reproducibility is strong: code is included, and the settings are well-documented.

### Weaknesses
1. GGMS assumes synchronous communication, identical deterministic state machines, and decisions in the final round. These simplifications are standard in this line of research. Still, showing one partially relaxed setting (e.g., I guess early decision or message subset) would strengthen the message that GGMS generalizes beyond toy cases.

---

2. The empirical section is minimal. Only a few experiments are tested, with few figures and no summary tables for success rates, runtime, or verification outcomes. The results are presented in qualitative line plots without statistical analysis or detailed metrics. More quantitative evidence or tabulated comparisons would make the evaluation more convincing.

---

3. Baseline comparisons are not strong enough: only MCTS and MCTS+DFS are tested, with no comparison to prior methods. Even a brief qualitative discussion or positioning would help readers place GGMS among existing work. This is my main concern. In addition, the analysis has limited insight: it mostly shows that GGMS, an intuitively promising integration of several modules, outperforms ablations that remove one or two modules.

---

4. The theoretical “provably correct” claim seems too strong. The two theorems are logically coherent within a finite-search setting (finite search space, accurate unfreezing), but they do not provide strong guarantees for practical learning or correctness, especially when stochasticity is introduced. The claim of “eventual convergence” could still imply astronomically long runtimes, as there is no complexity bound.

---

5. Minor issue: The figures could be improved. Figures 1–5 lack detailed captions, which makes them hard to understand.

### Questions
I don’t have many specific questions; my main concern is the experiments and insights behind them. Could the authors add qualitative comparisons with prior work, not only MCTS or MCTS+DFS, to clarify how GGMS differs in goals and outcomes?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper considers how to synthesize correct distributed protocols by combining Monte Carlo tree search, SMT-based correctness analysis, and specialized pruning procedures for guidance. Preliminary experiments and ablation studies are reported.

### Strengths
Applying learning for knowledge discovery is an exciting goal.

### Weaknesses
I do not find the motivation compelling at all. Distributed algorithms is a fascinating area where very smart people have contributed highly sophisticated algorithms over decades. It's unclear how the methods described in this paper can advance the area. Below are a few representative gaps between the toy setting used in the paper and distributed protocols: (1) use of FSMs to describe individual processes of a protocol is common, but one needs some form of symbolic descriptions that uses variables (of finite types) and guards and transitions over these. This is common in theory (e.g. see Nancy Lynch's 1995 book on Distributed Algorithms) and in protocol analysis tools (e.g. model checkers such as SPIN and NuSMV). (2) Process descriptions are symmetric but not identical (e.g. they make decisions based on comparison with Process ID). (3) Real challenges in protocol analysis are in implementation: high-level algorithms are typically published in a conference such as PODC with correctness proofs, and bugs show up in their translation to real-world platforms; contemporary work on protocol verification focuses on implementations (4) Protocols are typically analyzed using model checkers which can check both safety and liveness properties, SMT solvers are very limited since they can check only a small finite unrolling of the protocol (they are more scalable, but model checkers can easily analyze the toy state machines you are generating). In theory, one could  study the synthesis problem in a restrictive setting with the goal of removing these restrictions one by one, but the proposed approach does not inspire such confidence (for me).

### Questions
1. Why insist on "without human knowledge" and from scratch when this is an area with thousands of published papers?
2. If you are thinking a protocol that you would like to synthesize (and is not currently "known"), can you try asking a SOTA LLM (GPT-5) to produce it? It's not guaranteed to be correct, but can be checked by a model checker and counterexample can be used for prompting.

### Soundness
2

### Presentation
3

### Contribution
1
