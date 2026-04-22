# SocialJax: An Evaluation Suite for Multi-agent Reinforcement Learning in Sequential Social Dilemmas

- Avg Score: 5.20
- Decision: Accept (Poster)
- Scores: 4, 2, 8, 6, 6

## Abstract
Sequential social dilemmas pose a significant challenge in the field of multi-agent reinforcement learning (MARL), requiring environments that accurately reflect the tension between individual and collective interests.
Previous benchmarks and environments, such as Melting Pot, provide an evaluation protocol that measures generalization to new social partners in various test scenarios. However, running reinforcement learning algorithms in traditional environments requires substantial computational resources. 
In this paper, we introduce SocialJax, a suite of sequential social dilemma environments and algorithms implemented in JAX. JAX is a high-performance numerical computing library for Python that enables significant improvements in operational efficiency. Our experiments demonstrate that the SocialJax training pipeline achieves at least 50\texttimes{} speed-up in real-time performance compared to Melting Pot’s RLlib baselines. Additionally, we validate the effectiveness of baseline algorithms within SocialJax environments. Finally, we use Schelling diagrams to verify the social dilemma properties of these environments, ensuring that they accurately capture the dynamics of social dilemmas.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper implements a series of sequential social dilemma environments in JAX, each exhibiting mixed incentive dynamics and significantly improving the speed of environment simulation. A range of multi-agent reinforcement learning (MARL) algorithms are implemented, and training these algorithms in SocialJAX is at least 50 times faster than in MeltingPot2.0. The performance of various environments and algorithms is benchmarked in terms of both simulation throughput and training speed.

### Strengths
The main contribution of this article is the implementation of a series of reinforcement learning algorithms using JAX, along with experiments and a comprehensive analysis.
If these works can be open-sourced, they would have practical significance for scenarios in the industry where there is a high demand for intelligent agents.

### Weaknesses
1. The work presented in this paper involves implementing multi-agent algorithms in the field of sequential social dilemma environments using the JAX library, which has accelerated the execution speed of the algorithms. The focus seems to be more on engineering aspects rather than innovation. The paper does not claim to have made any specific innovative engineering contributions that would have sped up the algorithms during implementation. It appears that the performance improvements are largely due to the advanced features of the JAX library.

2. This work is not declared open-source, and the links provided in the abstract cannot be accessed. This might be due to time constraints or the intention to add more information during the rebuttal phase, but it is unfair to other works. Not being open-source limits the contribution to the community, and the inability to view the links makes it impossible to evaluate the workload.

### Questions
this paper may address its issues of innovation rather than providing more engineering descriptions. If it is an engineering contribution, more detailed information should be provided during the experimental phase.

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
5

### Summary
This paper presents SocialJax, a new suite of sequential social dilemma environments and algorithms built with JAX to advance research in multi-agent reinforcement learning (MARL).  In the second part of the paper, the authors validate the performance of baseline algorithms and use Schelling diagrams to study social dilemma outcomes - this second part appears rather disjointed from the first part.

### Strengths
+ Standardised benchmarks for studying social dilemmas using MARL agents are useful for the community. This paper goes in that direction.

### Weaknesses
- The discussion about CPU-based parallelism is difficult to understand. In fact, it seems that the problem is not at the level of the environment for GPU parallelism, but at the level of the agent implementation. It is possible to run multiple agents in a distributed fashion as well.

- In the introduction, the authors also claim that the social dilemmas are highly challenging environments. This appears to be a rather stretched claim. There are several environments that are actually much more complex than the simple dilemmas supported by SocialJAX.

- The authors appear to mix the problem of model implementation and environment when they discuss MAPPO, etc. (see paragraph entitled “Implementation of Algorithms in JAX” in the Introduction). In fact, you could release a very fast library implementing MAPPO, PPO, etc. instead.

- The authors say that the existing social dilemmas are restricted by CPU performance. This is not true. In fact, the models themselves can currently be run using GPUs in theory, but there is no inherent speed-up for them given their nature.  It seems that there is a mixing between agent implementation and interaction between agents in an environment. You can use standard PyTorch and use the GPU for implementing the agents. JAX allows you to do some clever synchronization in distributed training, not sure if it is useful/essential for the models that are described in this paper. 

- In any case, the authors do not provide a discussion of the actual implementation details that lead to a speedup with respect to a “CPU implementation”. Details about the implementation, parallelization, and synchronization are missing.

- The structure of the paper is unconvincing in itself. There is no reason to introduce social dilemmas (and the mathematical details) in Section 3. The paper is supposed to be about a simulator, not about the models that are well known and widely used by the community. The set of games implemented are also quite standard (perhaps that section could be moved to an Appendix). 

- In Table 1, the authors should have reported information regarding training, not random actions. The authors should have reported results for an increased number of CPUs. 

- It is unclear if the authors used the same GPU cores in the simulations as Agapiou et al. Also, these do not appear as meaningful comparisons given the fact that these are different training libraries (for example, it is not due to the implementation of Melting Pot, etc.).

- The discussion about the IPPO common reward is not relevant to this paper. It is a generic result orthogonal to the problem of the design of a MARL simulator.

- The results in Table 2 and 3 and Figure 3 appear unrelated to the goal of this work. These are unrelated to the evaluation of the simulator in itself. The analysis of the metric is also quite disjointed from the core goal of the paper.

- In the Appendix, the wall clock comparison is not meaningful since they are based on different implementations (some of them are also pixel-based).

Minor weaknesses:

- There are missing citations for certain games in Section 4.2.
- Arena is in bold in Section 4.2.

### Questions
- Can you provide some technical description about the design choices exploiting JAX characteristics in order to support GPU parallelism (also in comparison to the characteristics of the “standard” environments discussed in your paper)?
- Did you the same GPU cores as Agapiou et al? Are the performance figures taken from that paper?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces SocialJax, a JAX-based suite of nine sequential social-dilemma environments plus baseline MARL implementations (IPPO under individual/common reward, MAPPO, SVO, PPO-RE). The environments are grid-based abstractions inspired by Melting Pot (e.g., Coins, Commons Harvest variants, Clean Up, Coop Mining, Mushrooms, Gift Refinement, Prisoner’s Dilemma: Arena), with a fixed 11×11 partially observable window. The authors emphasize throughput and reproducibility: by having thousands of vectorized environments on GPU this paper shows significant speedups by using SocialJax that allow reproducibility with significantly less computational resources. They validate “social dilemma” structure via Schelling diagrams built from policy classes: “cooperators” sampled from common-reward training; “defectors” from individual-reward training, and propose simple, environment-specific cooperation metrics beyond return.

### Strengths
The paper is well written and introduces a valuable benchmark for the MARL community. A major bottleneck of MARL research is associated with the large costs of training often available only to big industrial labs. In such a way, SocialJax is a step towards the democratization of MARL research by allowing fast training of important benchmarks with significantly lower computational constraints. I believe the main strength of this paper lies on the opportunity it might enable to push the frontiers of MARL research by making it more affordable and thus it should be accepted to the conference.

### Weaknesses
A core weakness is that SocialJax reduces the observation resolution and modality relative to the original environments, shifting from pixel observations (e.g., Melting Pot’s ~88×88 image crops for an 11×11 field of view) to discrete 11×11 grid values rendered at the grid level. This removes the perception-learning burden that makes the originals attractive for testing MARL scalability and representation learning, and it changes coordination difficulty and partial observability in ways that can inflate apparent sample efficiency. In the authors’ own description, SocialJax agents “observe a grid matrix” with a fixed 11×11 view, whereas Melting Pot provides pixel images; even Melting Pot contest entrants who downsampled still operated in a pixel pipeline, not a hand-engineered grid state. Consequently, speedups and learning curves are not apples-to-apples comparisons.

**Minor errors/typos:**

**Line 271:** The authors state “we evaluate cooperation by counting how many coins each agent collects that match their assigned color”. This is not a valid metric of cooperation as the agents may be learning to collect all the coins. A good metric of cooperation in this environment is “total coins of their assigned color”/ ”total coins collected” i.e. which fraction of the coins collected are of their color.

**Line 192, 768:** Duplicated word “with with a beam”.

### Questions
How well do the different algorithm implementation performances align with those in the original environments (e.g. IPPO in meltinpot’s common harvest open vs socialjax's common harvest open)?

Some results are unintuitive to me. In the Schelling diagrams, why is the reward of defectors lower than that of cooperators in Gift refinement as the number of other cooperators increases? What are the policies that the cooperators are learning?

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
2

### Summary
The authors described an evaluation suite for multi-agent RL in social dilemmas.
This suite consisted of 9 RL environments exploring cooperation, partnership and trust
The authors also implemented SOTA algorithms to test the evaluation suite 
The key contribution is implementing the environment in JAX, which has a substantial improvement in rollout performance, directly speeding up training and evaluation of RL algorithms

### Strengths
1. The paper outlined the problem clearly
2. The authors presented a strong speed up improvement compared to the existing social dilemma environments
3. There is a diverse set of environments exploring many aspects of social dilemma
4. The authors provided a strong benchmark of RL algorithms for future comparisons

### Weaknesses
1. The authors clam to have a 50x+ speed up compared against the Melting pot 2.0. However, there were many differences in the observation space. It doesn't feel like a fair apples to apples comparison.  
2. The effects of changing the layout grid-like observation was not discussed
3. It might be helpful to understand how the same RL algo compares between the SocialJax vs Melting Pot 2.0

### Questions
* Why was the observation space changed? Is it possible to have a similar observation space for comparisons (possibly as an option)

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
SocialJax presents a suite of Melting Pot–derived tasks implemented in a lightweight minigrid framework to probe sequential social dilemmas among multiple reinforcement-learning agents. Built on JAX, the benchmark emphasizes end-to-end vectorization to significantly speed up both training and evaluation. To validate each environment’s social incentives, if they encourage individualistic or altruistic behavior, the authors use Schelling diagrams produced via the SVO algorithm. They also evaluate several multi-agent variants of PPO, establishing standardized baselines and illustrating performance differences across tasks. The environments were evaluated to be different from each other as the number of cooperators-individual payoff curves differ between environments. The learning curves against PPO models are also quite different and thought provoking how these variations in PPO impact altruistic behaviour so much.

### Strengths
* Clear, well-structured writing that’s easy to follow.
* Learning-curve plots are clean and legible, making comparisons straightforward.
* Variations among PPO architectures and their returns provide informative contrasts.
* Effective use of JAX for MARL: aligns with non-stationarity in transitions, rewards, and observations, and the performance benefits are evident.
* Insightful comparison of individually rewarded PPO vs. common-rewarded PPO, plus PPO-RE’s exchange schedule.
* Well-chosen environments that focus on mixed-motive cooperation which is an important aspect of multi-agent interaction and underexplored.
* Performance descriptions and the accompanying Schelling analyses are reasonable and generally supported.
* The Melting-Pot suite is theoretically grounded in game theory and the inclusion to JAX is welcomed

### Weaknesses
* Most environments are direct ports from Melting Pot or prior work; Coins and STORM already have JAX implementations, so the contribution feels more like technical consolidation than novel insight into mixed-motive RL.
* All evaluated algorithms are PPO variants; including other cooperative MARL approaches would better test generality and strengthen the conclusions.
* The Schelling diagrams (via SVO) are under-explained and lack context; it’s unclear why they’re preferable to simple metrics like cumulative individual rewards (for individualistic agents) and cumulative common rewards (for altruistic agents) or maybe even behavioural metrics. Related discussion to Schelling and why the Schelling diagrams are better is important
* Agent partial-observation windows are not visualized in the environment images, limiting clarity about information constraints.
* The benchmark is restricted to Melting Pot–based environments, which may introduce intra-suite bias and limit external validity.
* The study does not examine mixed individual and common reward settings, omitting an important class of real-world mixed-motive scenarios.
* Lack of explanation for why PPO-RE struggles in coop mining leaves the analysis incomplete.
* Possible exploration deficits are not investigated; lack of exploration may explain lower returns for some PPO variants.
* No off-policy or value mixer baselines

### Questions
I enjoyed reading the paper and following the results. That said, I was left wanting a deeper understanding of what’s new beyond Melting Pot, since the suite is derived from it. SocialJax’s focus on mixed motives (individual vs. altruistic behavior) is valuable precisely because it avoids the extremes of full competition or full cooperation. With that in mind:

* One classic mixed competition–cooperation task is Level-Based Foraging (LBF, [https://github.com/semitable/lb-foraging](https://github.com/semitable/lb-foraging)), where agents compete for reward but sometimes benefit from cooperating. Would you consider expanding the suite to include LBF or similar non–Melting Pot tasks to mitigate intra-suite bias and strengthen generality?

* The authors compared purely individual vs. purely shared rewards, but not mixtures. Prior work (Wang et al., *Individual Reward Assisted Multi-Agent Reinforcement Learning*, : [https://proceedings.mlr.press/v162/wang22ao.html](https://proceedings.mlr.press/v162/wang22ao.html)) shows that adding individual rewards to shared objectives in SMAC can hurt performance; recent directions in a similar vain explore higher-gradient adjustments ([https://arxiv.org/abs/2505.20579](https://arxiv.org/abs/2505.20579)) and value-shaping heuristics ([https://arxiv.org/abs/2508.17696](https://arxiv.org/abs/2508.17696)). SocialJax seems well-suited to probe this mixed-motive effect. Could the authors evaluate the IRAT method from Wang et al. on their suite since it is PPO-based, and report how well it performs?

* Could the authors add a few baselines with modified versions of their environments that introduce individually rewarded “distractor” objects (e.g., apples) that disappear and pay out when stepped on? This should induce local individualistic behavior; measuring how agents balance these against common goals could deepen the robustness analysis of altruism for the different methods presented.

* A key missing baseline is from the family of opponent-shaping algorithms. Proximal Advantage Alignment (“adalign”) from *Advantage Alignment Algorithms* ([https://arxiv.org/abs/2406.14662](https://arxiv.org/abs/2406.14662)) is a PPO-style method that steers policies via advantage signals and reportedly succeeds on Commons Harvest in Melting Pot. Would you include adalign to broaden algorithmic diversity and test how it's coordination compares in your tasks?

### Soundness
4

### Presentation
3

### Contribution
3
