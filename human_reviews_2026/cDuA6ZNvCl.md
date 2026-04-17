# Efficient Agent Training for Computer Use

- Decision: Accept (Poster)
- Scores: 4, 6, 2

## Abstract
Scaling up high-quality trajectory data has long been a critical bottleneck for developing human-like computer use agents. We introduce PC Agent-E, an efficient agent training framework that significantly reduces reliance on large-scale human demonstrations. Starting with just 312 human-annotated computer use trajectories, we further augment them by synthesizing diverse alternative action decisions with Claude 3.7 Sonnet. Trained on these enriched trajectories, our PC Agent-E model achieved a remarkable 141% relative improvement, and even surpassed the Claude 3.7 Sonnet by 10% in relative terms on WindowsAgentArena-V2, an improved benchmark we also released. By integrating robust human computer use skills with automated AI data synthesis capabilities, our method not only brought substantial improvements over training on human trajectories alone, but also significantly surpassed direct distillation from Claude 3.7 Sonnet.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces **PC Agent-E**, an efficient training framework for “computer-use” agents that aims to reduce reliance on large, costly human demonstration corpora. Starting from **312** human-annotated trajectories collected with a screen/interaction logger, the authors (1) reconstruct the missing step-wise reasoning via **Thought Completion**, and (2) use a strong teacher model (Claude 3.7) to synthesize **diverse alternative actions per state** (**Trajectory Boost**), yielding a dense, single-step supervision signal without online execution of the synthetic branches. A single agent is then trained end-to-end on the augmented trajectories.

On an improved benchmark the authors release, **WindowsAgentArena-V2**, PC Agent-E reports a **36.0%** overall success rate, **surpassing Claude 3.7 (with extended thinking)** and substantially improving over a Qwen2.5-VL-72B baseline (reported as **14.9% → 36.0%**, i.e., ~**141%** relative gain). The paper also shows positive transfer to **OSWorld**, suggesting cross-platform generalization. The project page and repository provide code, data, and deployment scripts, and outline the four-stage pipeline (collection → thought completion → trajectory boost → training).

### Strengths
* **Better benchmark signal (quality & significance).**
  Through human verification and fixes, the paper delivers a cleaner **WindowsAgentArena-V2** benchmark that reduces evaluation pathologies and provides clearer training/evaluation signals for computer-use agents.

* **Strong results with little data (originality & significance).**
  Despite limited human trajectories, the proposed training pipeline achieves **state-of-the-art performance on WindowsAgentArena**, allowing an open-source model to **match or surpass Claude 3.7 Sonnet**, highlighting a practical, data-efficient path for competitive open agents.

### Weaknesses
* **Efficient training lacks out-of-domain generalization.**
  While the method reaches ~35% on WindowsAgentArena (on par with Claude 3.7), it drops to **14.9% on OSWorld**, whereas **Claude 3.7 is ~35%** under a 50-step cap. This gap indicates the approach chiefly captures Windows-specific patterns rather than learning transferable skills—i.e., likely **in-domain overfitting** despite the “efficient training” claim.

### Questions
* **Evidence for “efficient training.”**
  Prior work (e.g., OpenCUA, UI-TARS) achieves capable agents mainly via large-scale data. Your claim that small-data, step-wise augmentation suffices is counter-intuitive. Could you provide stronger, targeted evidence that *efficiency*—rather than hidden data or compute—drives the gains?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces PC Agent-E, an efficient training framework for developing computer use agents with small amounts of human annotated data. The key idea of the paper is to use initial human demonstrations as seed data and bootstrap multiple trajectories demonstrating different ways of completing the same task synthetically by treating individual states from human trajectories as initial states and executing a powerful teacher agent starting this state. Through this approach we can spawn a computer use agent at intermediate steps of trajectories and roll it out to collect task execution trajectories that follow different paths. The paper shows this approach enables achieving state-of-the-art performance with minimal human annotation—just 312 human-demonstrated trajectories collected in a single day by two annotators.

The framework consists of 3 components:

1. Trajectory Collection: Gathering a small set of human computer use trajectories using PC Tracker, which records screenshots and keyboard/mouse actions for various Windows tasks.
2. Thought Completion: Reconstructing the implicit reasoning process behind each human action using Claude 3.7 Sonnet, converting raw action sequences into trajectories with explicit thought traces.
3. Trajectory Boost: Augmenting each trajectory step by sampling 9 alternative action decisions from Claude 3.7 Sonnet. This creates a "Traj Tree" where the human trajectory forms the trunk and synthesized alternatives branch as leaves, resulting in 27K training samples from the original 312 trajectories.

PC Agent-E achieves a 141% relative improvement over its base model and surpasses Claude 3.7 Sonnet by ~4% on WindowsAgentArena-V2, a benchmark the authors improved to address evaluation dependencies, infeasible task hacking, and other limitations. The model also generalizes to Linux environments (OSWorld) despite being trained only on Windows data.

### Strengths
1. The proposed approach is a nice way to train performant agents under limited data regime where we can collect small amounts of human demonstrations for each task and augment them synthetically to scale up demonstrations.
2. The additional contribution of a new evaluation benchmark WindowsAgentArena-v2 is valuable to the community and helps improve the evaluation benchmarks overall by fixing existing issues.
3. The analysis presented in section 5.3 is quite insightful. It highlights the fact that simply collecting “diverse” trajectories by high temperature sampling from initial state performs worse than spawning the CUA agent at intermediate states from a high quality demonstration which is quite interesting result.
4. Additional results on cross-platform evaluation are also interesting, demonstrate usefulness of the dataset, and are a valuable addition to the paper.

### Weaknesses
1. The caption of table 2 has a typo in difference between performance of teacher model Claude Sonnet 3.7 vs PCAgent-E. The text says the difference is 10% whereas it is ~4%. Authors need to fix this issue in the claims.
2. The test-time scaling results presented in section 5.5 seem incomplete to me. For main experiments in table 1 authors use max 30 step limit but in test time scaling experiments the two values use for step limit is 15 and 30 which seems counter intuitive. I request the authors to rerun the experiments with 50 step and 100 step limits to comprehensively evaluate whether the test-time scaling behavior exists in the finetuned agent or not. Eventhough current results demonstrate scaling from 15 to 30 steps it is unclear whether that is due to trajectory length being in distribution to the fine-tuning dataset or due to actual test-time scaling behavior.
3. It would also be good if authors can compare all methods in table 1 on WindowsAgentArena-v1 benchmark too so that we have a complete picture of how much improvement does the dataset lead to across both the setups and for fair comparison.
4. One question I have for authors is: It is interesting that simply scaling number of demonstrations for small amounts of unique tasks leads to massive improvements (~14 → 36) on WindowsAgentArena-v2 benchmark. Did authors do any analysis on identifying what is the primary cause for this improvement? Even though a single task can be completed in many ways I would expect the benchmark performance to saturate  as you scale the scaling factor (as in figure 7). However the figure shows improving trend for the PCAgent-E. I am curious to see if simply scaling demonstrations for this limited amount of tasks is leading to skill transfer across tasks or could it just be that some tasks are harder to learn and therefore require more demonstration instances per task. It would be a valuable analysis if authors can add to the paper as it will help understand the reason for gains better.
5. I would also like to see some analysis on task overlap for these 312 tasks with tasks in WindowsAgentArena-v2 so that we can understand how much train/test overlap is there.
6. There is a analysis section in 5.2 however I could not find the relative figure or table that analysis is talking about in the paper. It would be good if authors can either add it to supplementary or elaborate more on the failure mode breakdown and metrics in that section.

### Questions
Mentioned in Weaknesses section.

Overall I believe the contribution and experiments in the paper are interesting and valuable. However there are a few experiments and analysis that are missing or need to be added to make the paper stronger. I would be happy to increase my rating if authors address my concerns.

### Soundness
3

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
3

### Summary
Paper collects a dataset in a computer use domain and uses it to fine-tune QWEN with the goal of improving computer-use ability. It proposes TrajectoryBoost as a method to augment the few human demonstrations, and finds that their method outperforms direct distillation of Claude.

### Strengths
- Relevant problem of efficiently using small sets of expensive human data traces

### Weaknesses
- Evaluation only on a single and self-modified benchmark
- What about other benchmarks like OSWorld?
- Direct distillation from Claude not a fair baseline. It should be human data + direct distillation from Claude.

### Questions
see above

### Soundness
2

### Presentation
2

### Contribution
2
