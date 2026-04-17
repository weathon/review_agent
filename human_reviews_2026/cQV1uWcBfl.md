# Moving Out: Physically-grounded Human-AI Collaboration

- Decision: Reject
- Scores: 4, 6, 6

## Abstract
The ability to adapt to physical actions and constraints in an environment is crucial for embodied agents (e.g., robots) to effectively collaborate with humans.
Such physically grounded human-AI collaboration must account for the increased complexity of the continuous state-action space and constrained dynamics caused by physical constraints.
In this paper, we introduce Moving Out, a new human-AI collaboration benchmark that resembles a wide range of collaboration modes affected by physical attributes and constraints, such as moving heavy items together and maintaining consistent actions to move a big item around a corner.
Using Moving Out, we designed two tasks and collected human-human interaction data to evaluate models' abilities to adapt to diverse human behaviors and unseen physical attributes.
To address the challenges in physical environments, we propose a novel method, BASS (Behavior Augmentation, Simulation, and Selection), to enhance the diversity of agents and their understanding of the outcome of actions.
Our experiments show that BASS outperforms state-of-the-art models in AI-AI and human-AI collaboration.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Moving Out, a physically grounded benchmark for human-AI collaboration that emphasizes continuous action control and physical constraints. The benchmark includes two tasks supported by over 1,700 demonstrations. To address these challenges, the authors propose BASS, which augments collaborative trajectories, and simulates next states via a dynamics model under physical constraints. Experiments demonstrate that BASS outperforms other baselines in both AI-AI and human-AI collaboration, achieving higher task completion rates and better coordination.

### Strengths
1. This paper contributes a well-designed benchmark for symbolic human-AI collaboration and physically grounded interaction.

2. The BASS method integrates data augmentation and next-state prediction to enhance robustness to diverse human behavior and physical variations.

3. Extensive experiments, such as ablation studies, human user studies, and failure case analyses validate performance improvements.

### Weaknesses
1. Some key terms are not clearly defined, which reduces the overall clarity of the paper.

- L11: The phrase “adapt to physical actions and constraints” lacks precise definition. The authors should clarify what constitutes “physical actions” and “constraints.”

- L42: The scope of “diverse physical constraints, variations, and behavior” should be specified more precisely.

- L339, L341: The meanings of “aligned the boundaries” and “match the boundaries” are unclear.

- Title, Fig. 1, etc.: The authors emphasize a “physically grounded setting,” but the environments appear to be interactive simulations with limited rule sets (e.g., object mass, movement logic). The distinction between “physically grounded” and “grid world” is not well defined. If the intention is to highlight continuous action spaces, this should be stated explicitly.

2. The benchmark includes only two tasks (behavioral and physical generalization) with two agents. The limited diversity of agents and task types makes it unclear whether the benchmark captures broader aspects of collaboration, such as communication, planning, or anticipation.

3. In the Behavior Augmentation module, the recombination of sub-trajectories seems to involve exchanging trajectory segments between agents. The authors should clarify how this process improves action consistency.

4. Although the ablation study shows performance gains, it remains unclear how each submodule of BASS (augmentation, simulation, and selection) contributes to performance under different physical conditions or tasks.

5. The paper provides limited comparison with recent embodied collaboration frameworks. While HumanTHOR and Habitat 3.0 are mentioned, direct experimental comparisons are missing, which weakens the practical significance of the proposed approach.

### Questions
1. Some key terms are expected for clear definitions.

2. How do the proposed environments differ fundamentally from typical “grid worlds”? What specific properties make them “physically grounded”?

3. Given that the benchmark includes only two agents and two task types, do these settings generalize to broader aspects of human–AI collaboration, such as communication or planning?

4. How does sub-trajectory recombination improve action consistency between agents?

5. How each BASS submodule (augmentation, simulation, selection) contributes to performance under different conditions?

6. More collaboration frameworks such as HumanTHOR or Habitat 3.0 are expected to be compared.

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
This paper presents Moving Out, a human-AI collaboration benchmark, and BASS, a method for enhancing agent capability to handle human collaboration. The paper begins by motivating this problem and approach through the need for agents that can adapt to unseen human behaviors in collaborative settings. It outlines the high-level collaborations of a two-task benchmark, demonstration data for each task, and a novel data/training recipe for enhancing human-AI collaboration ability, consisting of components trained on this data. 

The paper goes on to formalize the problem as a decentralized MDP with two agents and discuss challenges of collaboration - unknown human policy distribution and large joint action and observation spaces. It then describes the environment and tasks. In the first task, focused on predicting and handling unseen human behaviors, models are trained on human-human collaboration demos, then tested with a new collaborator to see if they can generalize past seen behaviors. Evaluation is done by training multiple agents on splits of the data, then having the trained models collaborate with each other. The second task tests handling of physical attributes, particularly variation; evaluation is done by training agents on the full dataset and having them self-play. Demonstrations are collected from real humans as part of this work.

The paper then describes the augmentation method BASS. Partner poses are perturbed and subtrajectories are recombined to maintain coherence. A three-stage pipeline is built: an autoencoder to encode states, a dynamics model to predict the next latent state, and a second autoencoder to reconstruct the predicted state. Actions are chosen based on highest reward, where reward is a function of object distances to the goal region. Agents can then be trained on this data and select actions according to this framework. 

Finally, the paper reports on experiments. It poses four research questions: does BASS adapt to unseen behaviors, does BASS generalize to unseen physical constraints, does BASS improve effectiveness of collaboration, and failure patterns of BASS. Results show that Diffusion Policy (DP) with BASS is able to maintain the most performance across methods, though some others come close; DP with BASS outperforms baselines on certain key metrics measuring physical generalization; DP with BASS shows strong improvement over DP on human-collaborative tasks; and failure modes include various aspects of adapting to diverse behaviors, highlighting the inherent difficulty of high-level, conceptually diverse stimuli.

### Strengths
### Quality 
- Paper is very well-motivated 
- RQ structure is thoughtful. I appreciate not just that there are RQs, but that they target so many aspects of one concept, including failure modes. 
### Clarity 
- Well written
- Well-designed figures 
- Research question structure is very useful - helps understand what the paper is arguing. This is even more important given that the tasks, envs, data, etc. are expansive and somewhat arbitrary. Without the RQs and with just numbers, it would be hard to ascribe meaning to the results. 
### Originality and significance 
BASS seems novel, and the methods for constructing various aspects of the data are innovative and clever.


Overall, I think the benchmark is an interesting and useful contribution, even if there are related datasets/collection strategies and the design is similar to others. BASS is novel to my knowledge, but not particularly surprising given its reliance on the reward. Regardless, however, this paper is useful, and I think that warrants acceptance.

### Weaknesses
### Quality 
- Design is somewhat arbitrary. The tasks are very general, but their instantiations - the objects, environment, domain expansion techniques, etc. - are specific and feel somewhat arbitrary. Ultimately that might be fine, as all benchmarks have to be feasible to build and use, but it would help to have some justification for the particular instantiations.
- Evaluation lacks control. The numbers are convincing - BASS shows improvement on the tasks, and the tasks are large and expansive enough that I buy that that is meaningful. However, the evaluation is very high-level. Training AIs and then having them collaborate with humans, other AIs, or other AIs trained on this data - the behavior of such systems can have many explanations that are not necessarily differentiated by these experiments. 
- The results are strong, but they do rely on the reward signal, which adds an explicit addition over standard imitation learning. It's not surprising that that information helps.
### Clarity 
- No major concerns on clarity - the paper is put together well. 
### Originality and significance 
Human-AI collaboration is an established field. While BASS and the experiments done on it seem novel, the dataset is a very useful artifact but I'm not sure it's entirely novel. That said, since it is unique and useful, I don't think this is a major issue.

### Questions
I agree that the subtrajectory recombination is coherent. However, I suspect it deviates from naturalism. This may not matter - you don't need naturalism to have a good benchmark of complex behaviors, and it often only serves to add further complexity - but how do we think about the resulting distribution? What are some examples of pre- and post-recombination trajectories that build intuition for how/why BASS works?

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
2

### Summary
This paper presents Moving Out, a benchmark for physically grounded human-AI collaboration that addresses the challenges of continuous state-action spaces and physical constraints such as mass, shape, and friction. It introduces two core tasks—adapting to diverse human behaviors and generalizing to unseen physical attributes—and proposes BASS (Behavior Augmentation, Simulation, and Selection), a framework that integrates sub-trajectory swapping for data augmentation, a learned dynamics model for next-state prediction, and an action-selection mechanism to optimize actions under physical constraints. Extensive experiments and human-subject studies show that BASS significantly enhances task completion, action consistency, and robustness compared to strong baselines including GRU, MAPPO, and Diffusion Policy.

### Strengths
The Moving Out environment bridges the gap between symbolic multi-agent environments (e.g., Overcooked-AI) and physically grounded continuous control, with explicit physics and multi-modal collaboration types (coordination, awareness, action consistency).

The BASS framework introduces behavior recombination for partner diversity and a next-state simulation module for physical reasoning, which is well-motivated and conceptually solid.

Quantitative and qualitative analyses (Task Completion Rate, Normalized Final Distance, Waiting Time, Action Consistency) show clear improvements over competitive baselines.

### Weaknesses
The augmentation and simulation modules mainly combine known ideas (trajectory recombination, learned dynamics, and model-based scoring). The paper could clarify theoretical contributions beyond engineering integration — e.g., formal guarantees, diversity metrics, or physical constraint satisfaction proofs.

Both Task 1 (adapting to diverse human behaviors) and Task 2 (generalizing to unseen physical attributes) require human demonstrations or real-time human interaction. Although the authors introduce AI-AI self-play as a proxy evaluation, the true assessment of human-AI coordination still depends on human testing, which is labor-intensive and inconsistent.

Each human-in-the-loop evaluation requires participants to manually control agents, making experiments costly and variable. Behavioral differences among participants can significantly affect results, reducing reproducibility and preventing large-scale or cross-lab benchmarking.

### Questions
See Weakness.

### Soundness
3

### Presentation
3

### Contribution
2
