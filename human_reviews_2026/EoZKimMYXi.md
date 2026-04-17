# HA-VLN 2.0: An Open Benchmark and Leaderboard for Human-Aware Navigation in Discrete and Continuous Environments with Dynamic Multi-Human Interactions

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 8, 2, 2

## Abstract
Vision-and-Language Navigation (VLN) has been studied mainly in either discrete or continuous settings, with little attention to dynamic, crowded environments. We present HA-VLN 2.0, a unified benchmark introducing explicit social-awareness constraints. Our contributions are: (i) a standardized task and metrics capturing both goal accuracy and personal-space adherence; (ii) HAPS 2.0 dataset and simulators modeling multi-human interactions, outdoor contexts, and finer language–motion alignment; (iii) benchmarks on 16,844 socially grounded instructions, revealing sharp performance drops of leading agents under human dynamics and partial observability; and (iv) real-world robot experiments validating sim-to-real transfer, with an open leaderboard enabling transparent comparison. Results show that explicit social modeling improves navigation robustness and reduces collisions, underscoring the necessity of human-centric approaches. By releasing datasets, simulators, baselines, and protocols, HA-VLN 2.0 provides a strong foundation for safe, socially responsible navigation research.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes HA-VLN 2.0, a “unified” benchmark that integrates the discrete (DE) and continuous (CE) VLN settings under human-aware constraints; it defines a shared action set, introduces the upgraded HAPS 2.0 human-motion corpus, adds 16,844 HA-R2R instructions, and presents two baseline agents plus a leaderboard and real-robot demos.

### Strengths
The work tackles an important evaluation gap by releasing dual simulators with human motion, a unified API, and real-time multi-human rendering.

The real-robot section, while mostly qualitative, shows a full stack from sim to a Unitree Go2 with RGB-D/LiDAR, which lowers the barrier for follow-up work.

### Weaknesses
1. The authors have fundamental misunderstanding of the cocept of DE and CE. CE subsumes DE. “unifying” them as peers is misleading. CE supports fine-grained motion and collision-aware control, while DE is a graph-hop abstraction the authors themselves approximate via large step endpoints (“teleport-like”).
2. The paper claims a unified POMDP and “fair comparison,” but it specifies different sensing and motion models in DE vs CE, so the headline contribution does not hold at the problem definition level. DE results use panoramic RGB, CE agents use RGB-D, and CE collisions rely on radii overlap while DE hops among panoramic viewpoints—so observation and dynamics differ despite shared action names.
3. Dataset contribution is mostly synthetic augmentation. HA-R2R adds 16,844 instructions by prompting LLMs and extending R2R-CE, without reporting human agreement or grounding studies; this weakens the claim of a new, human-centric corpus addressing their stated gap.
4. Technical contribution is minimal. The two “baseline” agents are adaptations of existing models (VLN-BERT/Prevalent and CMA variants), so the work’s novelty rests on the benchmark story—which, as above, is not borne out. The experiment also lacks comparison with state-of-the-art models like Navid [1] and NaVILA [2].
5. The paper lists three “fundamental limitations” (social awareness, finer-grained instructions, dynamic crowds), but its fixes lean on LLM-generated text and do not establish actual crowded, multi-human complexity at the per-scene level; the 910 “active individuals” refers to dataset scale, not one scene.
6. The paper lacks a related work section in the main text. This makes understanding the position of the paper among related work difficult.

[1] Zhang, Jiazhao, et al. "Navid: Video-based vlm plans the next step for vision-and-language navigation." RSS 2024.

[2] Cheng, An-Chieh, et al. "Navila: Legged robot vision-language-action model for navigation." RSS 2025.

### Questions
What scientific insight is gained by framing DE and CE as one task when the observation function and transition model differ?

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces a unified framework for evaluating vision-and-language navigation (VLN) agents in both discrete and continuous settings that involve realistic, dynamic human interactions. It presents the HAPS 2.0 dataset, featuring rich multi-human motion and activity simulations, and the HA-R2R dataset, which adds socially grounded language instructions. The authors propose two baseline agents (HA-VLN-VL and HA-VLN-CMA) and demonstrate through extensive experiments—both in simulation and on real robots—that explicitly modeling human behavior improves navigation safety, social compliance, and robustness. Overall, HA-VLN 2.0 establishes a comprehensive benchmark for developing and evaluating human-aware embodied AI agents.

### Strengths
The paper is highly original in formulating a unified benchmark that bridges discrete and continuous vision-and-language navigation under dynamic, multi-human settings—a crucial yet underexplored dimension of embodied AI. Its quality is reflected in the rigorous design of datasets (HAPS 2.0 and HA-R2R), strong baseline models, and comprehensive evaluations across simulation and real-world deployment. The clarity of presentation is commendable, with well-structured methodology, consistent metrics, and transparent benchmarking. In terms of significance, the work establishes an important step toward socially intelligent navigation, providing a scalable, reproducible platform that is likely to influence future research on human-aware and safety-critical robotic navigation.

### Weaknesses
While the paper makes a strong contribution, several areas could be improved. First, the benchmark’s human behavior models, though diverse, rely on simulated motions rather than real human trajectories, which may limit ecological validity; integrating real-world motion capture data could enhance realism. Second, the evaluation focuses primarily on navigation success and collision metrics, with limited analysis of social acceptability or human comfort—key aspects for human-aware navigation. Third, the baseline models, though illustrative, are adaptations of existing VLN architectures rather than novel approaches specifically designed for social reasoning; incorporating explicit human-intent prediction or multi-agent interaction modeling could provide stronger baselines. Finally, additional ablations on the impact of instruction grounding and human density would clarify the sources of performance gains.

### Questions
1.How closely do the simulated human motions in HAPS 2.0 reflect real-world social dynamics? Are the motion sequences or interaction patterns derived from any real motion-capture or behavioral datasets?

2.The paper introduces human-awareness metrics such as personal-space adherence and collision rate—how were their thresholds and parameters determined?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents HA-VLN 2.0, a unified benchmark for vision-and-language navigation in dynamic, human-populated environments. It addresses socially compliant navigation by combining discrete and continuous settings within a single framework, enabling direct comparison of models trained under different paradigms.  
The work builds on an upgraded dataset, HAPS 2.0, which includes multi-human interactions, region-aware activity annotations, and semantically rich instructions grounded in social context. The simulator supports up to ten moving humans at real-time speeds and enforces human-aware constraints such as personal-space boundaries.  
Two baseline agents are evaluated against strong existing models using standardized metrics for navigation success and collision rates.  
The paper also introduces a public leaderboard and preliminary real-world experiments demonstrating partial sim-to-real transfer.  
Overall, HA-VLN 2.0 offers a promising benchmark and promising simulators for human-aware navigation research.

### Strengths
The paper addresses an important challenge in human–robot interaction: achieving socially compliant navigation in multi-human, dynamic environments. It introduces a unified benchmark for discrete and continuous environments, an area that remains underexplored.

- The proposed coupling of continuous and discrete environments into a shared-state representation allows direct and fair comparison across navigation paradigms.

- The inclusion of human-aware constraints, such as personal-space enforcement, is conceptually sound and likely beneficial for sim-to-real transfer.

- HAPS 2.0 represents a solid dataset contribution. The region-aware activity descriptions and multi-view annotations enrich the semantic content and improve realism.

- The human-focused instruction pipeline is well designed and clearly described.

- The simulation framework reportedly supports up to ten human agents at 30–60 FPS on standard GPUs. To the best of my knowledge, this level of performance is unmatched by other publicly available simulators.

Overall, the paper presents a promising and relevant addition to the field, particularly as a benchmark resource for socially aware robot navigation.

### Weaknesses
- Readability and organization :
The manuscript is difficult to follow due to excessive redirection to the appendix for core content. This disrupts the reading flow and fragments comprehension. A clearer separation between main content and supplementary details is needed, especially since a public project website already exists to host extended material. While the benchmark is the main contribution and its sim-to-real validation is a strong claim, Section 5.2, the section supporting these claims, is underdeveloped and lacks quantitative evidence. In contrast, ablation studies such as Table 4a occupy a big portion of the experimental discussion. Sections L197–205 and L234–242 provide low-level implementation details that could be moved to the appendix, given that readers are already frequently redirected there.

- Neglect of related work in Social Navigation :
Despite social navigation being central to this paper’s theme, the relevant state of the art is almost entirely omitted. Foundational works such as Cancelli et al. (2023), Puig et al. (2023–2024), and Scofano et al. (2025) are either missing or confined to the appendix. This is significant oversight, as the works are relevant to multiple sections of the paper:
    - Cancelli et al. (ICCV 2023), Puig et al. (ICLR 2024), and Scofano et al. (ICLR 2025, Spotlight) discuss safe and socially aware navigation in detail and are directly relevant to Sections L035-L047, L139, and L143.
    - The claim in L149 that this is the first “efficient simulation with realistic human-populated environments” is inaccurate, as Habitat 3.0 (Puig et al.) already provides such functionality.
    - PARTNR (Chang et al., ICLR 2025) should be cited as a relevant benchmark for realistic human–robot collaboration (L155–159).
    - The otherwise strong content in L253–256 would benefit from explicit comparison with Puig et al., which, as discussed by Scofano et al., struggles with multi-agent scalability.
Without a well defined comparison, the reader may be left wondering why this new benchmark is necessary when strong frameworks already exist.


- Experimental clarity:
The main experiments section (L372–385) lacks adequate explanation of the results in Table 1. The rationale for why proposed models (HA-VLN-CMA-Base, HA-VLN-CMA-DA, HA-VLN-CMA*, HA-VLN-VL) generally underperform relative to baseline models (BEVBert, ETPNav), except for HA-VLN-VL in the zero-shot setting, is unclear. Section 5.1 would benefit from more focused interpretation of these results.


- Real-world validation:
Section 5.2 is one of the paper’s most anticipated parts, as it relates directly to one of the major claims: real-world robot experiments validating sim-to-real transfer and the open leaderboard for transparent comparison. However, while L415-468 are well presented, the Real-World Validation & Setup subsection (L469-475) is incomplete. Although Figure 6b depicts a real-world experiment, no quantitative metrics (e.g., collision rate, success rate) are reported. To substantiate the sim-to-real claim, the authors should discuss observed limitations when transferring from simulation to the physical environment, including specific failure cases (e.g., collisions initiated by humans versus by the robot).


**Minor Weaknesses**

- Figures are overly complex and visually dense:
   - Figure 1 contains a paragraph-long caption that duplicates content from the main text (L069-072); it could be reduced to lines 072-075. The instruction text within the figure is unnecessarily long, numerical labels are barely visible, Finally, the bottom slider and different scenes do not aid comprehension. 
    - Figure 2 is similarly cluttered and difficult to read due to small fonts and visual overload.  
Both figures may benefit from redesign. 


- Missing citations:
SMPL is referenced in L086 and L099 without appropriate citations to foundational work.

### Questions
- In L073, the paper states: *“When the agent encounters a bystander on the phone, it intelligently turns right to avert a potential collision.”*  
What collision-avoidance logic is implemented for humanoid agents? Do humans always avoid collisions? In real-world contexts, humans can initiate collisions inadvertently. Is such behavior modeled? Prior work in social navigation allows humanoids to initiate contact to train agents to yield space. Do HA-VLN-DE and CE policies learn such reactive behavior?

- In the continuous-environment setting, the default step size is 0.25 m (L436–437). At what frequency does the robot collect observations and issue actions? Are human agents updated asynchronously while the robot is computing its next step?

- Could the authors elaborate on the results in Table 1?
The HA-VLN-CMA agent appears conceptually promising (L305-320, focusing on re-planning under obstacles), yet results in Table 1 show higher collision rates than baselines across all conditions. What could be the reason?

- What specific hardware is meant by “standard GPUs” (L255-256)? Figure A4 is referenced, but it does not clearly show the expected distribution of simulation speed versus the number of agents. Could such a distribution be provided?

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces a visual language navigation benchmark in the presence of dynamic human agents. It includes both discrete, and continuous environments, and common metrics between the two.

### Strengths
The paper provides an additional benchmark for VLN in human environments, and includes additional training data.

### Weaknesses
The main limitation of the paper is that it does not clearly articulate what the value of this proposed new benchmark is, in the light of previous ones.
There are at least two other very similar human VLN benchmarks, published in 2024:

Vuong, An, Toan Nguyen, Minh Nhat Vu, Baoru Huang, H. T. T. Binh, Thieu Vo, and Anh Nguyen. "Habicrowd: A high performance simulator for crowd-aware visual navigation." In 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pp. 5821-5827. IEEE, 2024.

Li, Heng, Minghan Li, Zhi-Qi Cheng, Yifei Dong, Yuxuan Zhou, Jun-Yan He, Qi Dai, Teruko Mitamura, and Alexander G. Hauptmann. "Human-aware vision-and-language navigation: Bridging simulation to reality with dynamic human interactions." Advances in Neural Information Processing Systems 37 (2024): 119411-119442.

The claim is that no previous benchmarks include dynamics, but the above two do claim to include dynamics.

Other details:
- The "continuous environment" also seems to be discretized, just with a slightly finer resolution. 
- The metrics seem rather adhoc, and not explained. For example, why is |A^c_i| subtracted from the collisions count? The two are different quantities entirely.

### Questions
- What is novel about this benchmark in light of the previous papers, and why is this difference (if any) significant?
- Is the continuous environment indeed discretized?
- Can you please explain the rationale behind the metrics?

### Soundness
2

### Presentation
3

### Contribution
1
