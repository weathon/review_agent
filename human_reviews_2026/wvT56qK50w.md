# Learning Goal-Oriented Language-Guided Navigation with Self-Improving Demonstrations at Scale

- Decision: Reject
- Scores: 6, 2, 4, 6

## Abstract
Goal-oriented language-guided navigation requires robust exploration capabilities for agents to navigate to specified goals in unknown environments without step-by-step instructions. Existing methods tend to exclusively utilize shortest-path trajectories, lacking effective exploration priors for training navigation agents. To address the above challenges, we present SID, a goal-oriented language-guided navigation learning approach with Self-Improving Demonstrations. Specifically, SID learns an initial agent on the shortest-path data sampled from environments and then leverages this agent to generate novel exploration trajectories. The novel rollouts provide demonstrations with stronger exploration strategies to train a better agent, which in turn produces higher-quality agent demonstrations for the next round of training. We show that this iterative self-improving pipeline readily scales to new environments, and the resulting demonstrations can be transferred across a variety of language-guided navigation tasks, elevating the performance ceiling in diverse goal-oriented navigation tasks. Extensive experiments demonstrate that SID significantly boosts the exploration capabilities and generalization of navigation agents. The resulting agent achieves new state-of-the-art performance on goal-oriented language-guided navigation tasks, including REVERIE, SOON, notably achieving a 50.9% success rate on the unseen validation splits of SOON, surpassing the prior leading approaches by a margin of 13.9%.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work studies navigation of an agent to a goal that is described by a language prompt. The method that the authors propose is an iterative self-distillation or self-training approach where a model initially learns to imitate the trajectory of a manual expert-crafted planning algorithm and then over multiple steps a new student is trained to imitate the trajectory of the previous model, mixed again with data of optimal shortest-path trajectories. The model is evaluated on 3 different simulation benchmarks.

### Strengths
- it is an interesting and relevant result that self-training is yielding such strong results. It is possible that the same finding applies to other tasks that are usually trained on priviledged information, because they all have a similar problem of briding the information gap between what the agent can observe and the information that is used for the training signal.
- I enjoyed reading this paper, the writing is easy to follow even for somebody who is not deeply familiar with this research bubble.

### Weaknesses
- To me this paper really seems the product of a research bubble that is a bit disconnected from reality. I missed widely known baselines from object-goal naviation such as [GOAT](https://arxiv.org/pdf/2311.06430) (unfortunately seems in this field there are 2 methods called GOAT), [UniGoal](https://openaccess.thecvf.com/content/CVPR2025/papers/Yin_UniGoal_Towards_Universal_Zero-shot_Goal-oriented_Navigation_CVPR_2025_paper.pdf), [SPOC](https://openaccess.thecvf.com/content/CVPR2024/papers/Ehsani_SPOC_Imitating_Shortest_Paths_in_Simulation_Enables_Effective_Navigation_and_CVPR_2024_paper.pdf), [VLFM](https://arxiv.org/pdf/2312.03275). Most of these methods are actually demonstrated to work on real robots, which goes far beyond this paper.  
Another indiation of this "bubble" are false statements in the paper that may apply to other imitation learning methods but not to the studied problem:
  - line 42: The authors state that in order to find an object, it is necessary that an agent explores the environment. That is incorrect. It is actually way more realistic that the agent has contextual information of a previous mission in the environment, and there are many methods investigating scene representations for language-prompted navigation such as [HOV-SG](https://arxiv.org/abs/2403.17846), [DynaMem](https://arxiv.org/abs/2411.04999), [osmAG-LLM](https://arxiv.org/abs/2507.12753).
  - line 159: "Existing methods rely on costly human demonstrations". Again incorrect. There are plenty of methods that do not require human demonstrations but instead rely on classical geometric methods for exploration, such as [VLFM](https://arxiv.org/pdf/2312.03275). There is even work on exploration without geometry and yet it does not require human demonstrations [FrontierNet](https://arxiv.org/abs/2501.04597)
- There are some critical pieces of information missing regarding the method and experiments:
  - It is not clear what the action space in line 212 is. I cannot tell from anywhere in the paper if this work reduces navigation to a 2D problem or if it could also navigate over multiple floors.
  - I am quite confused by the phrase "validation unseen". Are these environments used as validation set for hyperparameter tuning, early stopping, etc? Or are they unseen environments?
  - in Table 11 I am also not sure if these environments are seen or unseen during training. If unseen, why is only 1 baseline compared? Is there something about the setup that makes these numbers uncomparable with e.g. the numbers in VLFM?

### Questions
For the rebuttal I am looking forward to get a response to these questions:
- Can authors compare their results to more methdologically different approaches to language-prompted navigation or can they explain why it is not comparable?
- Can the authors discuss the applicability of their approach to actual real-world navigation problems? Are these advancements measured here (when e.g. compared to https://arxiv.org/pdf/2311.06430 expected to also lead to better real-world navigation?
- Can authors clarify the critical information explained above?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes Self-Improving Demonstration (SID), a strategy for goal-oriented Vision-and-Language Navigation (VLN). SID addresses the limitations of existing methods that rely solely on shortest-path trajectories, which lack effective exploration priors for training navigation agents. The authors first pretrain a base navigation agent using shortest-path demonstrations, then allow the agent to explore and collect additional successful trajectories. These successful runs are added to the training dataset, and the policy is iteratively retrained to progressively improve both data quality and agent capability. The method is evaluated on REVERIE and SOON benchmarks, achieving state-of-the-art performance on goal-oriented language-guided navigation tasks.

### Strengths
- The trained agent achieves state-of-the-art performance on the REVERIE and SOON VLN benchmarks.


- When combined with a vision-language model (VLM), SID enables the generation of 46M large-scale, language-goal exploration trajectories, providing valuable data diversity and scale.

### Weaknesses
The proposed method appears conceptually naive and closely resembles online reinforcement learning, with the main distinction that SID employs imitation-learning-based objectives. It is unclear why the authors avoid standard RL finetuning on top of the pretrained base navigation agent [1], which could likely achieve comparable outcomes using a simple sparse success reward (1 for success, 0 for failure). Moreover, the claim that RL requires “sophisticated reward engineering” (line 045) seems overstated in this context. 


That said, it is possible I have overlooked aspects of the method’s novelty—if the authors can provide clearer evidence of its unique contributions, I would be happy to raise my scores.

### Questions
Where can I find the section containing the “mathematical derivations supporting the theoretical claims,” as mentioned in the Reproducibility Statement? It does not appear to be included or referenced in the main text or appendix.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents a framework for goal-oriented language grounding that enables large language models to perform spatially and temporally coherent actions in embodied environments. It proposes a two-phase training approach combining semantic instruction alignment with trajectory-level feedback, allowing the model to generalize from high-level goals to fine-grained actions. Experiments on vision-language navigation and interactive tasks demonstrate improved goal success rates and interpretability compared to standard instruction-following baselines .

### Strengths
1. The authors propose a reasonable way to scale VLN data, and the collected dataset is of relatively large scale (~46M instances), surpassing existing datasets.
2. Benchmarks on multiple embodied datasets, showing consistent gains in both navigation success across benchmarks.

### Weaknesses
1. The idea is relatively widely explored in different domains, thus weakening the novelty of this work. Specifically, the general idea of using self-generated trajectories for SFT is a standard technique in LLMs and VLMs across language and vision-language tasks. While the paper does scale this approach in the VLN domain, the methodological contributions are limited.
2. The paper is only focused on vision-language navigation tasks, which is a somewhat niche application and it is unclear if the trained models can have applications in other domains.
3. The model architecture is still based on LXMERT, which is far behind state-of-the-art models and it would be good to see results with SoTA LLMs.

### Questions
Can your method be integrated with other types of agents, such as UI/visual tool-use agents?

### Soundness
2

### Presentation
2

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
The paper tackles goal-oriented language-guided navigation (e.g., REVERIE, SOON) and proposes Self-Improving Demonstrations (SID): start with shortest-path data, train a navigator, then let the navigator roll out in environments and keep only successful exploration trajectories as new demonstrations; iterate, then transfer trajectories to language settings via VLM-generated captions. This yields a massive corpus of ~46.5M language-goal exploration trajectories over ~860 environments and reports strong results on REVERIE/SOON, plus gains on ObjectNav.

### Strengths
1. Clear paradigm contribution. The data flywheel evolves the paths rather than the instructions, explicitly injecting exploration priors into imitation learning for goal-oriented VLN.
2. Scale and coverage. The authors assemble 46,534,355 trajectories (avg 9.01 viewpoints) across ~860 environments—an order-of-magnitude leap over prior VLN corpora oriented to instruction generation rather than exploration.
3. Solid empirical results. On REVERIE, SID reaches SR 59.4 / 60.4 (Val-Unseen/Test-Unseen) and SPL 46.0 / 45.9, outperforming recent large-scale baselines under a shared HM3D-800 setting.
4. Repro-friendly details. The paper provides practical implementation choices (e.g., DINOv2/SigLIP vision encoders; LXMERT init for DUET; InternVL2-26B for captions) and a GPU-hours breakdown across captioning, feature extraction, trajectory sampling, and pretraining.
5. Scaling analysis. The SID loop first converges on MP3D, then continues on 800 HM3D environments, maintaining prior skills while adding new exploration capability—evidence the scheme scales beyond a single dataset.

### Weaknesses
1. Caption dependency and alignment. Transferring trajectories to language tasks leans on a VLM (InternVL2-26B) to annotate images. The paper would benefit from systematic quality and bias analysis of captions and their impact on downstream SR/SPL. 
2. Cost and accessibility. The method includes non-trivial compute for captioning and repeated rollout/sampling; while the GPU-hours table is appreciated, the paper doesn’t deeply examine sample-efficiency vs. performance (e.g., marginal gains per SID round).
3. Failure modes & generality. The training/evaluation are in indoor, panoramic, graph-based simulators; it’s unclear how well the learned exploration prior translates to continuous control or real-world deployment without navigation graphs. (SIM-to-REAL discussion could be expanded.)
4. Ablations are implied but could be deeper. The paper emphasizes that evolving trajectories (vs. more shortest-paths) drives the gain, yet a more granular ablation—e.g., isolation of “teacher-forcing from exploration demos” vs “student-forcing with oracle,” or trajectory-quality stratification—would strengthen the causal story behind SID’s improvements (especially on SOON). (Results tables are strong; still, mechanism-level ablations could be expanded.)

### Questions
1. Caption robustness: How sensitive are results to the style (detail/REVERIE/SOON prompts) and factuality of captions? Any automatic or human audits? 
2. Value of each SID round: What are the marginal gains vs. compute per round on MP3D and during HM3D scaling? Could early stopping save most of the compute without noticeable loss?
3. Transfer beyond indoor graphs: Have you tried a continuous-action simulator or partial topological maps to probe generality?

### Soundness
2

### Presentation
3

### Contribution
3
