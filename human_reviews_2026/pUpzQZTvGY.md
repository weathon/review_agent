# Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents

- Decision: Accept (Poster)
- Scores: 4, 4, 8, 6

## Abstract
Most of today's AI systems are constrained by human-designed, fixed architectures and cannot autonomously and continuously improve themselves. The scientific method, on the other hand, is a cumulative and open-ended system, where each innovation builds upon previous artifacts, enabling future discoveries. There is growing hope that the current manual process of advancing AI could itself be automated. If done safely, such automation would accelerate AI development and allow us to reap its benefits much sooner. This prospect raises the question of how AI systems can endlessly improve themselves while getting better at solving relevant problems. Meta-learning can automate the discovery of novel algorithms, but is limited by first-order improvements and the human design of a suitable search space. The Gödel machine proposed a theoretical alternative: a self-improving AI that repeatedly modifies itself in a provably beneficial manner. Unfortunately, proving that most changes are net beneficial is impossible in practice. We introduce the Darwin Gödel Machine (DGM), a novel self-improving system that iteratively modifies its own code (thereby also improving its ability to modify its own codebase) and empirically validates each change using coding benchmarks. Inspired by Darwinian evolution and open-endedness research, the DGM grows an archive of generated coding agents. It samples agents from this archive, which self-modify to create new, interesting versions of themselves. This open-ended exploration forms a growing tree of diverse, high-quality agents and allows the parallel exploration of many different paths through the search space. Empirically, the DGM automatically improves its coding capabilities (e.g., better code editing tools, long-context window management, peer-review mechanisms), increasing performance on SWE-bench from 20.0% to 50.0%, and on Polyglot from 14.2% to 30.7%. Furthermore, the DGM significantly outperforms baselines without self-improvement or open-ended exploration. All experiments were done with safety precautions (e.g., sandboxing, human oversight). Overall, the DGM represents a significant step toward self-improving AI, capable of gathering its own stepping stones along a path that unfolds into endless innovation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces the Darwin Gödel Machine, a system designed for the open-ended evolution of self-improving AI coding agents. The core idea is to merge two powerful concepts: self-referential code modification and population-based, open-ended exploration. Unlike the theoretical Gödel Machine, which requires formal proofs of improvement, DGM uses empirical validation on coding benchmarks to guide its evolution. The system maintains a growing archive of all generated agent variants, allowing it to select parents for modification from a diverse pool of "stepping stones". Empirically, DGM demonstrates a significant performance increase on SWE-bench (from 20.0% to 50.0%) and Polyglot (from 14.2% to 30.7%). Ablation studies confirm that both the self-improvement mechanism and the open-ended exploration are essential for this sustained progress.

### Strengths
1. The performance gains on two distinct and challenging coding benchmarks are substantial and compelling. The ability to automatically discover sophisticated improvements is a demonstration of the system's efficacy.
    
2. The paper's primary strength is the successful and novel synthesis of self-referential modification with population-based open-endedness. While these ideas exist in other contexts, their combination here creates a virtuous cycle: better agents become better at creating even more capable offspring, and the evolutionary framework prevents the process from easily getting stuck.

### Weaknesses
1. The paper's claim to novelty in the self-referential aspect needs clearer differentiation from prior work. The paper does cite Yin et al. (2024) for the Gödel Agent and notes that in DGM, the downstream task (coding) directly aligns with the self-improvement task (also coding). However, this distinction is brief. A more detailed discussion is needed to clarify what specific limitations of the Gödel Agent's self-reference framework the DGM overcomes, or how its implementation of the concept is fundamentally different and more advantageous beyond the stated task alignment. Without this, the contribution appears more incremental than foundational in this specific dimension.
    
2. The DGM agent is not fully self-referential. The DGM agent can modify its own coding logic, but the open-ended exploration process itself (i.e., archive maintenance, parent selection strategy) remains fixed and human-designed. This limits the system's full autonomy, as it cannot learn to improve how it explores and evolves. The paper acknowledges this as future work, but it is a major limitation of the current system.
    
3. The paper is transparent about the high computational cost (an estimated $22,000 for a single SWE-bench run), which is a significant barrier to reproducibility and further research by the academic community. While the results are impressive, the cost may limit the practical applicability and exploration of this method.
4. What is the signal from the environment to support improvement? Is a validation dataset required for self-improvement?

### Questions
1. What are the anticipated challenges in allowing the DGM to modify its own open-ended exploration process (e.g., the parent selection mechanism in Appendix C.2)? Would this require a separate reward signal, or could it emerge from the existing objective?
    
2. Can you discuss potential scenarios where the core assumption of task-capability alignment might break down? For example, could an agent over-specialize in solving small bug fixes (like in SWE-bench) in a way that makes it less capable of performing large-scale architectural refactoring on its own codebase?
    
3. The analysis highlights the lineage of the final best agent, which traverses some lower-performing nodes. Does an analysis of "dead-end" evolutionary branches offer any insights into common failure modes or deceptive local optima in the agent design space?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes Darwin Godel Machines (DGMs) which is a system for self-improving AI except that it uses a fixed LLM (which they refer to as a frozen FM). DGM keeps an archive of agent variants and uses a parent-selection scheme to choose agents to self-modify. Each child is empirically evaluated on coding benchmarks and, if functional, added back to the archive. Empirical results are presented.

### Strengths
They got their novel algorithm to work, with clear signs of self improvement. Ablations show that parts of their algorithm are helping. Authors plan on releasing code.

### Weaknesses
1. The whole setup is the same as Recursive Self-Improving Code Generation (RSICG, Zelikman et al 2023b) which this paper cites, and the paper should be presented as a follow-up and improvement over that work. Both works consider the idea of recursive self-improvement with a fixed FM. 
2. Empirically, it should be compared to that work. (Side note about the name: aren't Godel machines more about provable improvements?) 
3. The paper repeatedly claims its main novelty is RSICG specifically:
> "the first self-improving system powered by FMs with open-ended exploration, where progress on its evaluation benchmarks can directly translate into better self-improvement capabilities"
What does this mean and how does it go beyond Zelikman et al? 
4. The main paper does not contain much detail at all about the algorithm, which is mainly described and defined in the appendix.
5. The ethical risks of RSI are not discussed. But clearly, the development of RSI poses potential risks that numerous luminaries claim are existential. See [https://superintelligence-statement.org/](https://superintelligence-statement.org/) for example. The discussion around safety focuses on the fact that the this paper's experiments were run in a sandbox, but there wasn't much concern that the system in the paper is superintelligent. The real risk is that it's advancing science towards that goal without clear discussion of why the benefits of this progress outweigh the risks.

### Questions
Is this RSICG and, if so, why is it not presented in that light? What is the novelty in this paper? Why is it not compared empirically to Zelikman et al, whose code is available online? Can you define the algorithm in the paper?

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
5

### Summary
DGM is a self-improving system that autonomously evolves its own codebase. It maintains an archive of coding agents that iteratively self-modify and empirically validate improvements on coding benchmarks. Through this process, DGM becomes increasingly proficient at both solving coding tasks and performing future self-improvements. Empirical results show substantial gains on SWE-bench and Polyglot.

### Strengths
Strong motivation. It is reasonable to expect that AI research will eventually be automated. This paper addresses an important direction likely to remain relevant in the long term.

Clear positioning. The connection to both meta-learning and the Gödel machine helps situate the contribution within established frameworks for self-improving systems.

Useful Ablations. The ablations are particularly useful, as they clearly demonstrate the contribution of different components of the proposed algorithm.

Original Ideas. The paper introduces a novel formulation of self-improving AI that integrates meta-learning, evolution, and recursive self-improvement.

### Weaknesses
Baselines are missing. The baselines used for comparison are also designed by researchers (they are ablated versions of the proposed method), and no direct comparison is made against other self-improving or open-ended systems in the literature. Adding such a baseline would strengthen the empirical claims. 

Only two benchmarks. SWE-bench and Polyglot are used in evaluation. Including an additional benchmark could help validate the generality of the approach.

Economic considerations. It is unclear whether better-performing agents come with proportionally greater inference costs. Theoretically, performance could improve simply by sampling n responses in parallel and increasing n at each generation step, especially in verifiable software engineering tasks. An analysis comparing the discovered agents to trivial test-time scaling (best of n) would help establish whether the improvements are meaningful rather than inefficient increase of computational costs.

see https://arxiv.org/pdf/2407.01502

The paper states: “Our framework envisions agents that can rewrite their own training scripts (including training a new foundation model (FM)). However, we do not show that in this paper, as training FMs is computationally intensive and would introduce substantial additional complexity, which we leave as future work.” 

While theoretically plausible, the economic and computational cost of such a setup can make large-scale application economically unjustifiable.

### Questions
- Do better agents necessarily incur higher inference costs? Could you plot a Pareto frontier that includes these inference costs, and are the discovered agents Pareto-optimal?
- Is DGM without self-improvement equivalent to ADAS, and DGM without open-ended exploration equivalent to STOP (Self-Taught Optimizer)? Please discuss their similarities and differences.
- If DGM can be viewed as combining ADAS and STOP with evolution, what is the benefit of reframing it as a Darwin–Gödel Machine?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies how LLM based coding agent system can autonomously improve by editing their own code base. The approach proposed in this paper improves on a line of similar work with the key introduction of a tree-structured archive of past coding agent implementations paired with a parent selection rule which the authors refer to as an "open-ended exploration loop". The authors evaluate their method on subsets of SWE-Bench and the Aider Polyglot benchmark, and demonstrate a good improvement from their first agent iteration to their last.

### Strengths
The paper is for the most part clearly written and does a good job of setting out the broader vision for self-improving agent systems and the promise that open-endedness holds. However, a lot of space is used up in the abstract and introduction for this vision-setting, sometimes introducing concepts which aren't returned to later. Perhaps more of the main body of the paper could be devoted to what is in my opinion the paper's main contribution, which is the "open-ended exploration" loop and details of the parent selection mechanism, which becomes somewhat lost in the main text, with key details pushed into the appendix. Indeed, the results in Figure 6 in the appendix show that this is the key enabler of the DGM's performance

### Weaknesses
The initial model performance (as shown, for instance in Figure 2 at iteration 0) is surprisingly low. Claude 3.5 Sonnet (new) [scored 49%](https://www.anthropic.com/news/3-5-models-and-computer-use) on the full SWE-Bench Verified benchmark using Anthropic's internal harness, while o3-mini (medium) [scores 53.8%](https://aider.chat/docs/leaderboards/) on Aider (albeit with pass@2 instead of pass@1 as used in the paper). What explains the seemingly large 29% and 39.8% drop, respectively, between the initial agent at iteration 0 used in the paper and the Anthropic and Aider scaffolds used with the same models? While I understand that Aider is a more mature scaffold which may explain the larger difference (and you may have used the old instead of the new benchmark so my 53.8% number may be wrong), I believe the agent scaffold used by Anthropic for 3.5 Sonnet (new) was fairly minimal. It might appear that the nature of the early improvements implemented by the DGM merely relate to fixing basic functionality in the agent loop. Moreover, it does not seem that the DGM (at not-insignificant monetary cost, Appendix E.1), has demonstrated an improvement beyond allowing the base model (3.5 Sonnet or o3-mini) to perform at its usual level of approximately 49% and 53% on SWE-Bench and Polyglot respectively, by fixing the initial scaffold.

The paper also only presents experiments on models released around the end of last year. In the intervening time, model labs have continued training models, improving their performance as coding agents by performing large-scale RL training with environments resembling simple agent scaffolds. For instance, the "bash only" [SWE-Bench leaderboard](https://www.swebench.com/) or recent papers such as [Dai et al., 2025](https://arxiv.org/pdf/2509.25873) demonstrate that modern models placed in minimal agent loops are outperforming those placed in more elaborate agent scaffolds. The corollary is that while agent scaffolds can support weaker models and guide them to perform tasks such as editing files or navigating codebases which they were not trained to do, as the foundation models are trained to perform these skills natively, such agent 'scaffolding' can become obsolete or even detrimental to agent performance. As a result, the paper's results and the relevance of its claims about being able to improve coding agents by editing their own codebase would be strengthened if they were re-validated with, for instance, Claude 4.5 Sonnet and GPT-5-thinking. Taking the SWE-Bench experiments for example, these should be started with the minimal [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) which already achieves 70.6% on SWE-Bench Verified. It would be valuable to demonstrate that the kinds of agent scaffold improvements DGM is able to implement continue to lead to meaningful performance improvements with these modern models. It is possible that with a more sophisticated base model, the types of scaffold improvements may themselves become more sophisticated and abstract; moving beyond simple editing mechanisms and the like towards task management and other more abstract functions. However this has yet to be shown.

In light of the above, it seems that simply modifying the agent's codebase limits the "action space" and precludes achieving the continuous and open-ended recursive self improvement discussed in the introduction. While I note that fine-tuning the weights of the agent's model has been acknowledged by the authors as something for future work, I do believe that the paper would have been strengthened by examining this approach here as a first instance: potentially on a much simpler task than SWE-Bench and with a much smaller foundation model.

Finally, my reading of the transferability results of the DGM-discovered agent across different benchmarks and programming languages at the top of page 8 is less optimistic than the authors put it. The results seem to indicate slight over-fitting to the task the DGM self-improves with, rather than broad-based improvements to the coding agent's capabilities. What is however interesting is that SWE-Bench appears to be a better "teaching" task than polyglot, with the Polyglot-trained agent doing comparatively poorly on SWE-Bench, whereas the SWE-Bench trained agent almost matches the performance of the Polyglot-trained agent on Polyglot.

Nits:
- L243: The line reading "We evaluate the DGM on two popular benchmarks" reads awkwardly, given the discussion of SWE-Bench and Polyglot immediately prior.

### Questions
- Why is the baseline (iteration 0) performance of the initial agent quite low (20%) on SWE-Bench initially?
- Where does the 'representative agent baseline (Aider)' number come from on the right pane of Figure 2? Is this the score for o3-mini on Aider's *old* code editing benchmark?
- Looking at the curvature of the light blue 'Average of Archive' line in Figure 3, it appears to be negative from iteration 50 onwards. What do you think is the cause of this plateau, and does this point to a limitation in the open-ended improvement loop (perhaps new idea generation, or the FM's coding ability, or something else entirely)?
- What is the variance within each run? For example, in Figure 2, at times the green line (DGM without self-improvement) is ahead of the full DGM's blue line. While I understand the significant time and monetary costs of doing each run, if each experiment were to be repeated 100 times, do you think this trend would continue? In particular, what prevents over-specialization of the agent to the proxy training task (i.e. solving narrow issue-resolution tasks in SWE-Bench's style) which may in time preclude effective self-improvement?

### Soundness
2

### Presentation
3

### Contribution
3
