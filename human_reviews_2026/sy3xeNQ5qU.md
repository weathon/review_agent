# Adversarial Policy Transfer in Mixed Cooperative-Competitive Games

- Decision: Reject
- Scores: 4, 4, 6

## Abstract
Equilibrium learning in mixed cooperative–competitive games remains a central challenge, as empirical algorithms often become trapped in suboptimal or locally stable equilibria. Adversarial policies expose vulnerabilities in such equilibria learned by victim agents through task-irrelevant actions, a problem well studied in two-agent zero-sum games and only recently extended to multi-agent reinforcement learning (MARL). Existing approaches often overfit to specific scenarios and lack generalization, considering task-specific vulnerabilities only and requiring millions of interactions to adapt to new settings, which is impractical
in real world. By contrast, transferable MARL methods can generalize across tasks but focus on overpowering opponents rather than strategically exploiting their weaknesses.
Here we propose a transferable adversarial policy framework for mixed cooperative–competitive games that enables zero-shot attacks in previously unseen scenarios, revealing the existence of shared vulnerabilities in learned MARL policies and enabling efficient and accurate robustness assessment without training a separate attack for each policy. Our approach has two key components. First, \emph{adversarial tactic acquisition} iteratively extracts attack strategies that reliably deceive victim agents, using large language models (LLMs) during training and Bayesian inference to weight tactics at test time. Second, \emph{adversarial scene decomposition} partitions attack scenarios into smaller, transferable subgames that consistently elicit adversarial behaviour, based on the interactions between attacker and victim teams. We provide a convergence proof alongside our approach.
Empirically, we demonstrate adversarial policy transfer in \emph{StarCraft II} and \emph{MAgent} across 20 tasks with up to 64 victim agents, varying in number, type and policy. Training against our attack addresses common vulnerabilities in victim policies and enhances robustness to subsequent re-attacks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates whether it is possible to learn transferrable adversarial attacks in the multi-agent cooperative/competitive setting, where agents are cooperating within a team and competing across teams. They propose two key methods for learning transferable attacks: having a LLM infer a set of common tactics across a set of successful trajectories (and training the actors to follow those), and breaking down a game into subgames (in terms of which other players to pay attention to) and using those subgames to more easily find commonalities with shared scenarios in other unseen games. They show that they are able to learn substantially more effective adversarial policies than other baselines.

### Strengths
- The paper is clear and well written, and does a good job laying out its goals and focuses, as well as clearly articulating its main method 
- The methods used are innovative and interesting, and are potentially applicable across a broad range of RL scenarios

### Weaknesses
- I felt confused about what parts of the learning or optimization process were applied on each new environment, vs whether all learning was only taking place at "training" level and no optimization was happening on new environments. (For example: is it the case that the LLM tactic generation is fixed at train time, but at test time there is optimization performed to better align with the LLMs dynamically generated labels of how actions map to tactics? 
- I felt confused about what made these attacks intrinsically adversarial, or intrinsically focused on the cooperative/competitive scenario. I can't tell if this was just one scenario they happened to explore, or if this method is specifically designed for this paradigm in a way I'm not following

### Questions
- What makes the attacks you are learning inherently adversarial? It seems like the kinds of behaviors described (regroup, retreat, etc) are simply strategic moves within the game, which is naturally adversarial in that it is a competitive game on the team level, but I don't understand what differentiates the method described here from one that would work just as well for figuring out generally good transferrable strategies 
- Do these approaches of having LLMs summarize tactics work for types of games beyond the specific competitive/cooperative type analyzed here? I was unclear on whether the latter was actually an important part of the applicability of this method, or if it was simply one example of where it could work 
- Is any form of learning or optimization (including aligning actions to desired tactics) performed on new environments? Or are action-level policies fully fixed once you hit unseen environments? 
- I think there are a lot of good things about this paper, but I would want to get answers to these questions before being more confident of an "accept" review

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes Adversarial Policy Transfer (APT), a new framework for improving policy transfer and generalization in multi-agent reinforcement learning (MARL).  Instead of directly fine-tuning source policies in new target environments, APT introduces an adversary agent during training that intentionally challenges or perturbs the main policy. The learner minimizes performance degradation under these adversarial perturbations, leading to a more  transferable policy.  

The approach combines:
1. Adversarial optimization to expose weaknesses of the source policy,  
2. Policy distillation to stabilize knowledge transfer, and  
3. A unified transfer objective for both cooperative and competitive MARL settings.  

Experiments on Multi-Agent Particle Environments (MPE), Overcooked-AI, and Atari two-player games demonstrate that APT improves transfer performance and generalization to unseen partners or opponents.  Some experiments also include LLM-driven agents for testing robustness against diverse or human-like behaviors, though LLMs are not part of the APT learning process itself.

### Strengths
- The paper presents a conceptually novel extension of adversarial learning to policy transfer.  
  The idea of using adversarial agents not for equilibrium learning but for transfer is well-motivated and new within the MARL context.  

- The formulation of a bi-level adversarial objective, combined with policy distillation, is technically sound and well-justified.  

- The experiments are solid. APT consistently outperforms fine-tuning, MAICL, and ATT across both cooperative and competitive benchmarks.  

- The paper includes thorough ablation studies, examining fixed vs. adaptive guidance, the number of adversaries, dataset size, and shared vs. separate opponent models.  

- The writing is clear, well-structured, and easy to follow.

### Weaknesses
- The novelty is incremental. The framework combines known ideas—adversarial training, policy distillation, and transfer learning—into a coherent and well-executed system.  

- The analysis justifies the adaptive adversarial weight locally (via score disagreement) but lacks global convergence or stability guarantees.  
  There is also no discussion on local-global consistency, which is a key consideration in cooperative MARL.  

- The paper does not include a rigorous proof of convergence for the APT algorithm.  
  The bi-level optimization between the adversary and the learner is described conceptually and algorithmically, but there is no theoretical analysis regarding convergence to equilibrium, stability of adversarial updates, or consistency of the transfer process.  

- The evaluated tasks (MPE, Overcooked, Atari) are relatively small and low-dimensional.  
  Including experiments on more challenging MARL benchmarks (e.g., SMACv2) would better demonstrate scalability and robustness.  

- LLMs are used only to generate partner or opponent behaviors in certain evaluations, and there is no ablation study to assess their contribution to overall performance.

### Questions
1. How does APT differ from previous combinations of adversarial training and transfer learning in MARL and single-agent RL?  
2. Can the authors provide any theoretical insights or guarantees regarding convergence or stability of the bi-level optimization?  
3. What is the impact of LLM-based agents on the reported results, and can this be isolated through ablation?  
4. Does the adaptive adversarial weighting mechanism maintain consistency between local and global objectives in cooperative settings?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a method combining LLM and Bayesian reasoning for adversarial policy transfer in multi agent reinforcement learning. Policies are modeled as probabilistic embeddings, with LLM generating dynamic strategy labels during training. Using a CTDE framework and Transformer backbone, the approach enables zero shot transfer to unseen tasks, aiming to enhance adaptability and generalization without fine tuning the LLM.

### Strengths
This paper provides a valuable exploration of combining large language models with Bayesian inference and effectively enhances agents’ ability to adapt strategies in complex environments. The proposed method is innovative, enabling the integration of natural language understanding into policy learning, which broadens the research perspective in this field. The overall structure of the paper is clear, and the technical details are supported by solid derivations and arguments. In addition, the experimental data and analysis are persuasive, thoroughly demonstrating both the strengths and limitations of the method.

### Weaknesses
The author uses LLMs to extract common paradigms from successful attack cases, but the paper does not provide detailed information about the sources or the sufficiency of these cases. Furthermore, the subsequent strategy iterations based on LLM tags rely entirely on the strategies given by the LLM in the first round, with only the weights of different tags being adjusted. Does this reliance on the initial strategy risk weakening the exploration of the strategy space and potentially lead to getting stuck in a local optimum?

### Questions
See weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3
