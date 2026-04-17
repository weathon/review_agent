# STAIRS-Former: Spatio-Temporal Attention with Interleaved Recursive Structure TransFormer for Offline Mulit-task Multi-agent Reinforcement Learning

- Decision: Accept (Poster)
- Scores: 8, 4, 4, 6

## Abstract
Offline multi-agent reinforcement learning (MARL) with multi-task datasets is challenging due to varying numbers of agents across tasks and the need to generalize to unseen scenarios. Prior works employ transformers with observation tokenization and hierarchical skill learning to address these issues. However, they underutilize the transformer attention mechanism for inter-agent coordination and rely on a single history token, which limits their ability to capture long-horizon temporal dependencies in partially observable MARL settings. In this paper, we propose STAIRS-Former, a transformer architecture augmented with spatial and temporal hierarchies that enables effective attention over critical tokens while capturing long interaction histories. We further introduce token dropout to enhance robustness and generalization across varying agent populations. Extensive experiments on diverse multi-agent benchmarks, including SMAC, SMAC-v2, MPE, and MaMuJoCo, with multi-task datasets demonstrate that STAIRS-Former consistently outperforms prior methods and achieves new state-of-the-art performance.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper introduces STAIRS-Former, a transformer-based architecture for offline multi-task multi-agent reinforcement learning (MT-MARL). It addresses two main limitations in prior transformer-based MARL models (like UPDeT, ODIS, and HiSSD): poor handling of long-term temporal dependencies and limited relational reasoning among entities. STAIRS-Former introduces three modules — (1) a spatial recursive transformer for deeper inter-agent correlation modeling, (2) a dual-scale temporal module that maintains short- and long-term histories, and (3) a token-dropout mechanism to improve robustness across varying numbers of agents. Extensive experiments on SMAC benchmarks (Marine-Easy, Marine-Hard, and Stalker-Zealot) show consistent and significant improvements over state-of-the-art baselines, with ablation studies and interpretability analyses (e.g., attention and dormant neuron studies) supporting the architectural design

### Strengths
This paper makes a clear and well-motivated contribution to the field of offline multi-task multi-agent reinforcement learning. The authors identify real and important limitations in existing transformer-based MARL models—specifically their difficulty in capturing long-term temporal relations and rich inter-agent dependencies—and address them with a framework that feels both technically sound and intuitively designed. The proposed STAIRS-Former architecture is an elegant combination of spatial recursion and dual-scale temporal modeling, allowing the system to reason jointly over history and agent relations in a way that existing models cannot.

The experimental section is particularly strong: the evaluation on multiple SMAC benchmarks is thorough, ablation studies are comprehensive, and the visualizations provide genuine insight into how the model learns. The paper is also very well written and organized, making it easy to follow both the motivation and the technical details. Finally, the work feels significant because it pushes transformer-based MARL toward more scalable and generalizable architectures, offering a practical foundation for future research in multi-agent decision-making systems.

### Weaknesses
While the framework is strong and the results are convincing, the paper would benefit from a clearer discussion of generalization beyond the SMAC environment. SMAC’s discrete and tokenizable observation space makes it naturally suited to transformer architectures, so it’s uncertain whether STAIRS-Former would maintain its advantages in less structured or multi-modal domains (e.g., visual-linguistic inputs or real-world sensor data). A brief evaluation or qualitative analysis in such settings would significantly strengthen the paper’s claim to generality.

### Questions
Generality beyond SMAC: Have you tested or considered applying STAIRS-Former to environments with more complex or unstructured observations, such as multi-modal inputs (e.g., visual or continuous sensor data)? If not, how do you anticipate the model’s spatial recursion and tokenization scheme would adapt in such cases?

### Soundness
4

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
2

### Summary
The paper introduces STAIRS-Former, a spatio-temporal transformer architecture for offline multi-task multi-agent reinforcement learning (MT-MARL). The proposed method includes (1) a novel spatial recursive model to extract correlations among local observations of different entities, and (2) a novel temporal module that helps mitigating partial-observability in MARL settings and allows for capturing long-term dependencies.
The authors carry out experiments on offline SMAC v1 datasets in multi-task fashion, showing improved performance over previous baselines. Particularly, STAIRS-Former displays impressive generalization over unseen tasks, as well as varying number of agents, likely due to the token dropout mechanism.

### Strengths
- The authors tackle a relevant and underexplored setting (offline MT-MARL) which is likely of interest to the community and opens important directions of application.
- The paper shows clear improvements and strong empirical performance over relevant baselines in the field (UPDeT-m, ODIS, HiSSD), specifically over unseen tasks.

### Weaknesses
- Limited and outdated benchmark tasks: the authors solely present their comparison in the context of the SMAC benchmark. The authors neglected experimentation on more recent benchmarks such as the improved benchmark SMACv2 [1], as well as the MaMuJoCo benchmark [2]. Considering the nature of this work is mostly empirical, and that SOTA online MARL methods notoriously test on these benchmark, it is unclear why the authors only provide experimentations on SMAC v1. In turn, it's unclear how the architectural contributions in STAIRS-Former really compare against relevant benchmark tasks in the field.

[1] Ellis, Benjamin, et al. "Smacv2: An improved benchmark for cooperative multi-agent reinforcement learning." Advances in Neural Information Processing Systems 36 (2023): 37567-37593.

[2] Peng, Bei, et al. "Facmac: Factored multi-agent centralised policy gradients." Advances in Neural Information Processing Systems 34 (2021): 12208-12221.

### Questions
- Why was SMACv1 chosen by the authors against the improved SMACv2 benchmark?
- Please clarify how the tasks are divided into training and tests, with respect to number of agents and different goals/rewards. It appears to me that the authors do not rely on explicit task-conditioning information at training time, so I'm assuming agents must implicitly infer the task from observations. If so, how can they generalize to a task with a different objective? Or do tasks only differ by the amount of agents?
- The authors claim in the abstract that a Transformer module is not able to capture long-range dependencies  because it compresses the entire history into a single token. However, this is in general a false claim, because that's exactly what transformers claim to do over RNNs. Could you please clarify what is the main drawback of previous methods and whether the limitation over long-horizons is a fundamental consequence of the architecture itself or rather a resulting effect?

### Soundness
2

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
The paper targets offline multi-task multi-agent RL (MT-MARL) with varying agent counts, inputs and actions across tasks. It builds on UPDeT/transformer-style architectures and proposes (i) a “spatial recursive”/deeper transformer to get less uniform attention, (ii) a dual-timescale temporal/history module (short- and long-term), and (iii) token dropout to generalize across different token/entity counts. Experiments on SMAC multi-task offline datasets show improvements over UPDeT-m, ODIS, and HiSSD.

### Strengths
1. Problem setting (offline + multi-task + variable agents) is relevant for MT-MARL and aligns with recent transformer-based MARL lines.
2. Paper is clearly written and well-situated w.r.t. UPDeT/ODIS/HiSSD.

### Weaknesses
1. Overall, the paper integrates known ingredients rather than introducing a genuinely new architectural principle for offline MT-MARL.
2. If the problem is “uniform attention” in UPDeT/HiSSD, there are other mechanisms: attention sharpening, entropy regularization, auxiliary supervision on heads, or stronger positional/task conditioning. The paper should justify why a relatively heavy spatial–temporal–recursive stack is preferable to these lighter alternatives.
3. Baselines may be underpowered: The main UPDeT-style baselines in the paper use very shallow transformers (as the authors themselves note “one-layer transformer cannot capture diverse relations”). A fairer test is: what happens if we (i) increase the number of transformer layers, (ii) add a simple recurrent/history token with longer horizon. Right now, the improvement could just be due to “more depth + a GRUi.e., model capacity, not the specific STAIRS interleaving.
4. Experiments on other benchmarks such as MAMuJoCo, WareHouse, etc. can also be presented to enhance the experimental evaluations to compare how this method compares against other offline MARL baselines.

### Questions
1. Offline pretrained transformer-based MARL (MADT) show that one big sequence model can handle multiple SMAC tasks and benefit from offline pretraining. How does this method improve upon MADT and similar baselines?
2. For offline MARL, optimizing just the TD3-loss with BC regulation has shown to yield poor results because of very weak regularizations on the exploding joint action spaces. How has that been tackled here? Why did the authors use this method of training over existing offline MARL framework? 
3. For the comparisons with other methods, how many layers were used for the baselines vs the STAIRS?

### Soundness
2

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
The main challenge in offline multi-agent reinforcement learning (MARL) with multi-task (MT) datasets is the varying number of agents across tasks, which changes the input structure. Prior transformer and hierarchical skill methods underutilized the transformer's attention mechanism, focusing instead on transferable skills, and crucially, they suffered from poor historical context: they compressed the entire history into a single token at each step, making them function like a basic recurrent neural network that largely ignores long-term historical information despite its criticality in partially observable MARL. The proposed STAIRS-Former addresses this by augmenting the transformer with spatial and temporal hierarchies to effectively leverage long history and properly attend to critical tokens, while a new token dropout technique is incorporated to improve generalization to diverse agent populations; experiments on the StarCraft Multi-Agent Challenge (SMAC) benchmark confirm that STAIRS-Former achieves new state-of-the-art performance.

### Strengths
- I believe this paper is well-structured and written.
- The central problem of this work is well-motivated and does sound.
- I think this work offers a good solution for the realization of MT-MARL problem with the corresponding challenges. 
- Another strength point is the ablation on the algorithmic decision by the authors and the informative discussion.

### Weaknesses
- A crucial weakness of this work is not stating the limitations.
- I believe a limitation of this work could be the potential overhead and memory footprint due to the introduced components. Although, the overall training time is highlighted in the appendix, a deeper analysis would be appreciated where the training time or process time for each introduced component. This can be done by reporting the ablated training time if available or simply the overhead proccessing time compared to normal training step.
- In the experimental section, there is no highlighting for the model sizes used across methods. This could result in an unfair comparison to the baselines.
- Figure 5 is unclear. A more informative caption would be appreciated.

### Questions
- What are the limitations of this work?
- What is the memory footprint of the model and the introduced overhead compared to the other baselines?
- In Figure 6, what is "("wo RT" excludes repeat & TSFFN)"?
- I did not understand well the analysis in Figure 5. Would you mind elaborating more and clarify the heatmaps in the figure?

### Soundness
3

### Presentation
3

### Contribution
3
