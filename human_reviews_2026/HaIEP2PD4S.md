# WebFactory: Automated Compression of Foundational Language Intelligence into Grounded Web Agents

- Decision: Accept (Poster)
- Scores: 6, 4, 6

## Abstract
Current paradigms for training GUI agents are fundamentally limited by a reliance on either unsafe, non-reproducible live web interactions or costly, scarce human-crafted data and environments. We argue this focus on data volume overlooks a more critical factor: the efficiency of compressing a large language model's (LLM) latent knowledge into actionable agent behavior. We introduce WebFactory, a novel, fully automated closed-loop reinforcement learning pipeline for GUI agents, systematically compressing LLM-encoded internet intelligence into efficient, grounded actions. Our pipeline features a process of scalable environment synthesis → knowledge-aware task generation → LLM-powered trajectory collection → decomposed reward RL training → systematic agent evaluation.
Remarkably, our agent demonstrates exceptional data efficiency and generalization. Trained on synthetic data from only 10 websites within WebFactory, it achieves performance comparable to GUI agents trained on same amount of human-annotated data from a much larger set of environments. This superior performance is consistent across our internal offline and online transferring benchmarks, where our agent also significantly outperforms the base foundation model.
We further provide critical insights into the "embodiment potential" of different LLM foundations, offering a new axis for model evaluation. This work presents a scalable and cost-effective paradigm for transforming passive internet knowledge into active, grounded intelligence, marking a critical step towards general-purpose interactive agents.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces WebFactory, an automated, closed-loop pipeline for converting an LLM’s largely descriptive web knowledge into executable GUI/web-agent policies. The key design choice is to avoid the instability of the live web by operating in a fully controlled offline clone of real sites. The system (1) programmatically generates tasks from site knowledge, (2) uses a strong LLM to collect successful trajectories, and (3) trains an RL agent over a unified action space with decomposed rewards. The training is done on synthetic trajectories from 10 sites, and the resulting agent matches or has gains over agents trained on similarly sized human data, and shows non-trivial transfer to real platforms. The authors interpret this as a form of intelligence compression and propose LLM embodiment as a complementary axis for comparing foundation models.

### Strengths
- The paper tackles an important problem of grounding agents to real website with cheap and controllable data
- The idea is simple and easy to build upon
- The empirical result are strong and the gains are significant

### Weaknesses
The "compression" language seems odd to me. It makes sense to claim that training the model on data generated in a sim/controllable environment enables the model to generalize to a more realistic environment, but this compression framing makes it more confusing than necessary. There's already some early work related to sim2real that discuss how to use envs that are cheap and controllable to transfer to real env. I can see in some way of defining "compression" that stresses the knowledge part, but in that case more ablation/analysis should be included. E.g., Table 1 shows a breakdown w.r.t. the generation quality (executability, validity, etc) but not final RL agent performance.

The other main weakness is the lack of ablations. Out of the 10 websites which contributes most to the final performance? How does the performance curve look like as you add from fewer to more websites? How fast does the performance plateau (i.e., what's the diminishing return of adding envs)?

### Questions
See the weakness section for my main questions.

In the context of automatic pipelines that turn weak/indirect web knowledge into trajectories, run training, and test on real env, there are some references that I think should be discussed as well:
- WebShop: Towards Scalable Real-World Web Interaction with Grounded Language Agents
- Synatra: Turning Indirect Knowledge into Direct Demonstrations for Digital Agents at Scale
- AgentTrek: Agent Trajectory Synthesis via Guiding Replay with Web Tutorials
- Explorer: Scaling Exploration-driven Web Trajectory Synthesis for Multimodal Web Agents

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper contributes a data generation pipeline for end-to-end training of GUI computer use agents that (1) constructs synthetic websites with diverse layouts and control interfaces, (2) populates these websites with synthetic listings for products / other materials, and (3) employs knowledge of the raw website product / listing data + the navigation structured between pages to efficiently generate verifiable web navigation tasks with a known optimal completion paths. 


The proposed pipeline is employed as an environment for training GUI agents derived from GUI-R1, and results show that RL on synthetic websites and tasks from the proposed pipeline can be as effective as training on human data (ie, human demonstrations, and human-written tasks).

### Strengths
## Originality:

There has been an increasing focus on solving the data problem for computer use agents in recent literature, and this is a problem of major importance in my opinion. To my knowledge, prop works have engaged largely with existing websites on the internet in the construction of trajectories and tasks, existing human-created benchmarks like WebArena. The problem statement of generating synthetic websites is original, and potentially a valuable tool for targeting the data distribution to models’ weaknesses.


A large focus of the paper’s motivation is on developing a scalable pipeline for task / environment  generation, but the discussion of current efforts could be more thorough. For example, there have been significant efforts this past year in creating computer use tasks with verifiable evaluation criteria on live websites that are not mentioned in the discussion, and must be considered when evaluating originality, such as:

* [1] Synatra: Turning Indirect Knowledge into Direct Demonstrations for Digital Agents at Scale, Ou et al. 2025
* [2] Harnessing Webpage UIs for Text-Rich Visual Understanding, Liu et al. 2022
* [3] Proposer-Agent-Evaluator(PAE): Autonomous Skill Discovery For Foundation Model Internet Agents, Zhou et al. 2024
* [4] NNetNav: Unsupervised Learning of Browser Agents Through Environment Interaction in the Wild, Murty et al. 2025
* [5] InSTA: Towards Internet-Scale Training For Agents, Trabucco et al. 2025

The paper would be improved by incorporating these citations into its discussion where relevant.

To my knowledge, generating synthetic websites is an original contribution, and has the potential to allow the pipeline to target weaknesses in the learner, which is a valuable capability not explored in this work.

## Quality:


Overall the paper is organized logically, and most important experimental questions are answered by the end. Relevant datasets are selected from the literature, and several baseline models are included that are relevant.

## Clarity:


The paper is generally written clearly, though some minor details could be improved, mentioned in the next section.

## Significance:


As discussed in my review of the paper’s originality, the notion of distilling language intelligence into agentic capabilities is not itself a new idea in the realm of web agent research [1-5]. In addition, several previous research methods have been scaled to extract tasks / trajectories from millions of websites [1,2,5] and can be considered scalable approaches for this reason. Therefore, the significance of this paper derives from the capability of the proposed system to generate specific targeted data with rich layouts, specific verifiable tasks with known optimal completion paths, and the ability to be deterministic (important for benchmarks).

### Weaknesses
While the reported numbers are impressive---an improvement of 162% over the Qwen VL baseline in Table 4 is large and a good validation of the approach---it is not clear in this experiment how tasks are created in this evaluation, and the number of websites + tasks is low (90 according to the experimental details). This number is low enough that I would be interested to see error bars (standard deviation, and 95% confidence interval) reported for this experiment in order to improve my confidence that the results are indeed this much better. 

In addition, if the tasks employed in Table 4 are not from a standard benchmark, but are instead created by the authors for the purposes of evaluating their agent, the methodology employed for task creation should be included in the paper, with a discussion of the steps taken to ensure that there is no bias introduced by potentially designing tasks in a way that favor their approach, which would make the results less trustworthy.


Finally, the experiments miss an interesting capability of the proposed method: the ability to probe the weaknesses of current GUI agents and generate websites that target those weaknesses. Since this is a fairly sophisticated suggestion that is unlikely to be feasible during peer review, this weakness is mainly as a suggestion for improving the paper, and not one that impacts my score.

### Questions
In Table 2, are the statistics reported for the same set of tasks, or are the task being varied (ie, statistics of trajectories for tasks generated with the Knowledge config vs without the Knowledge config)?


Line 346 reports that the decrease in steps is due to more efficient task execution, but if the tasks are being varied in this experiment, this could also be explained by a shift in the task distribution.


Which set of tasks are used in Table 4, and how are these selected? Are these from a standard dataset, or generated by the proposed method (important to know to ground the results here)?

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
This paper introduces WebFactory, a pipeline for automating the training of GUI-based web agents. The core idea is to use a high-fidelity, offline environment to synthesize tasks, generate trajectories with a strong LLM executor, and train an agent via reinforcement learning with a decomposed reward function. The authors claim this approach achieves superior data efficiency and generalization, with an agent trained on only 10 synthetic websites performing competitively against agents trained on larger sets of human-annotated data. The work also introduces the concept of "LLM embodiment" as a new axis for model evaluation.

### Strengths
I believe this paper propose some perspectives beyond a method to train the web agent. For example, paradigm shift from fine-tuning LLM to treating the LLM as embodiment. In this way, the LLM could be treated beyond the traditional SFT or RL paradigm. 

Besides, there are some open-source contributions. Authors are releasing the entire toolchain, including the 10 high-fidelity offline environments, task generators, and training pipeline. This is a engineering contribution that enables reproducible research and provides a high-quality, non-chaotic testbed for the community, distinct from the "noisy live web".

### Weaknesses
There is a contradiction in motivation and scalability. The introduction opens by attacking the reliance on "costly, scarce human-crafted data and environments" and explicitly calls out the "painstaking, manual synthesis of high-fidelity environments" as a bottleneck that "can itself consume weeks of expert effort". However, the entire WebFactory method is predicated on exactly this: a suite of 10 high-fidelity offline websites, which Appendix C reveals were meticulously built by the authors as a "Next.js/React monorepo". The "fully automated"  pipeline only functions after this massive, non-scalable, human-expert-driven environment creation step is complete. This is a glaring contradiction. The paper does not solve the environment synthesis bottleneck; it leverages it.

In this way, the claim of a "fully automated" pipeline is misleading. The automation only applies to data generation within the pre-built, sanitized, and manually-instrumented environment. This is a far from a system that can be pointed at website.

While the agent outperforms the baseline on the online transfer task (Tab. 4), its absolute performance is still low. An average TCR of 53.4% means the agent fails its task nearly half the time on real-world sites like Amazon and Airbnb. This highlights a major sim-to-real gap that the 10 pristine, static, React-based environments  fail to capture from the chaotic, non-deterministic, and constantly changing live web.

### Questions
Please refer to the weakness part.

### Soundness
3

### Presentation
4

### Contribution
3
