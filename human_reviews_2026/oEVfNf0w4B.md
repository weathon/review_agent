# ComputerRL: Scaling End-to-End Online Reinforcement Learning for Computer Use Agents

- Decision: Accept (Poster)
- Scores: 4, 6, 4

## Abstract
We introduce ComputerRL, a framework for autonomous desktop intelligence that enables agents to operate complex digital workspaces skillfully. ComputerRL features the API-GUI paradigm, which unifies programmatic API calls and direct GUI interaction to address the inherent mismatch between machine agents and human-centric desktop environments. Scaling end-to-end RL training is crucial for improvement and generalization across diverse desktop tasks; however, it remains challenging due to environmental inefficiency and instability during extended training. To support scalable and robust training, we develop a distributed RL infrastructure capable of orchestrating thousands of parallel virtual desktop environments to accelerate large-scale online RL. Furthermore, we propose Entropulse, a training strategy that alternates reinforcement learning with supervised fine-tuning, effectively mitigating entropy collapse during extended training runs. We employ ComputerRL on open models GLM-4-9B-0414 and GLM-4.1V-9B-Thinking, and evaluate them on the OSWorld benchmark. The GLM-ComputerRL-9B achieves a new state-of-the-art accuracy of 48.9%, demonstrating significant improvements for general agents in desktop automation. Our code is available at https://github.com/THUDM/ComputerRL.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces ComputerRL, a large-scale framework for training computer use agents that can operate desktop environments via a unified API-GUI interaction paradigm. The framework integrates (1) a distributed RL infrastructure supporting thousands of parallel virtual desktops, (2) an automated pipeline for generating APIs from LLMs to augment GUI control, and (3) a training strategy named Entropulse, which alternates between reinforcement learning (RL) and supervised fine-tuning (SFT) to mitigate entropy collapse during long training runs.
The proposed system is evaluated on the OSWorld and OfficeWorld benchmarks, achieving state-of-the-art results (48.9% success rate) on computer automation tasks with open-weight models.

### Strengths
The API-GUI unification is an interesting engineering contribution that bridges the gap between human-designed interfaces and agent-level programmatic control.

The distributed RL infrastructure is impressive in scale and demonstrates strong engineering capability, enabling parallelized desktop environments at large scale.

The Entropulse idea addresses an important issue in long-horizon RL (entropy collapse), and the empirical results suggest measurable benefits in maintaining exploration and training stability.

The evaluation across multiple benchmarks (OSWorld, OfficeWorld) provides strong empirical evidence of performance improvements.

### Weaknesses
The paper’s novelty lies primarily in implementation and scaling, not in new algorithmic contributions. The API-GUI paradigm is conceptually straightforward—it effectively automates API construction via LLMs rather than introducing a new interaction or reasoning mechanism. Similarly, the Entropulse training alternation between RL and SFT is more of a practical training schedule than a novel learning algorithm.

The training curves in the figures appear to correspond to single runs, with no error bars or indication of variance across seeds. It is therefore unclear how stable the reported improvements are. The lack of information about the number of runs or random seeds undermines the reliability of the reported trends.

The paper does not include prior literature on diversity or exploration in RL, which would be relevant given the focus on entropy restoration and SFT alternation. Existing works in exploration-based RL, policy diversity, or ensemble-based approaches could provide valuable context, but they are not mentioned.

Given the large-scale infrastructure described (thousands of virtual desktops and 9B-parameter models), the paper lacks concrete information on hardware setup, training cost, and total compute used. For a work that emphasizes scalability, this omission is significant. Details such as training duration, GPU/CPU utilization, and cluster size should be clearly reported to assess feasibility and reproducibility.

### Questions
Could you clarify what you consider the core algorithmic innovation of ComputerRL beyond the engineering scale-up?

You mention that Entropulse increases exploration and behavioral diversity — could you provide quantitative evidence (e.g., entropy statistics, trajectory variance, or action coverage) supporting this claim?

Could you provide details on the hardware setup used — such as GPU type/count, CPU cluster size, memory, and network bandwidth?

How does your approach compare to prior work addressing entropy collapse or exploration in RL (e.g., maximum entropy RL, population-based methods)?

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
This paper presents **ComputerRL**, a framework for training computer use agents capable of performing desktop operations through a unified API-GUI interaction paradigm. The framework integrates:
1. a distributed RL infrastructure that supports training across thousands of virtual desktop instances;
2. aan LLM-driven module that automatically derives application programming interfaces (APIs) to extend and complement traditional GUI-based controls;
3. Entropulse, a hybrid training regime that periodically alternates between RL and SFT phases to counteract entropy collapse and maintain exploration over extended training trajectories.

Evaluations on OSWorld and OfficeWorld benchmarks show state-of-the-art results (48.9% success rate) using open-weight models (GLM-4-9B-0414, GLM-4.1V-9B-Thinking), outperforming both proprietary and open baselines such as OpenAI CUA, Claude 4.0, and UI-TARS.

### Strengths
- **Strong systems and engineering contribution:** The distributed RL infrastructure is technically impressive, enabling large-scale online RL across thousands of virtualized desktop environments. Such scale is rare in open research and represents a substantial engineering achievement.

- **Practical API-GUI paradigm:**  The unified action space combining GUI operations with automatically constructed APIs addresses a key bottleneck in desktop automation. The LLM-driven API construction pipeline is pragmatic and lowers the barrier for generalization.

- **Entropulse training strategy:** Alternating RL and SFT phases effectively combats entropy collapse and stabilizes long-horizon training. While conceptually simple, it appears empirically effective and easy to adopt in practice.

- **Empirical performance:** The results on both OSWorld and OfficeWorld are strong and consistent. The proposed approach achieves superior performance and sample efficiency (fewer steps per task) compared to all evaluated baselines. Ablation studies indicate the importance of multi-stage training and the API-GUI design.

- **Writing and organization:** The paper is very well written, clearly structured, and visually well presented.

### Weaknesses
- **Unsubstantiated claims about diversity and exploration:** The paper claims that alternating SFT with RL increases exploration and diversity, yet no quantitative evidence is provided. Metrics such as action entropy, trajectory variance, or coverage are not analyzed. The only evidence is a qualitative entropy curve, which is insufficient.

- **Incomplete empirical rigor and reproducibility:** All training curves appear to represent single runs without confidence intervals or variance estimates. The number of seeds, randomization strategy, or statistical robustness is not discussed. Given the scale (9B-parameter models, thousands of desktops), compute and reproducibility details are critically missing — no information on hardware setup, training duration, or cluster configuration. For a paper emphasizing scalability, this omission is significant.

- **Limited methodological novelty:** The paper’s main innovations are engineering-oriented. The API-GUI paradigm is conceptually straightforward since it. The novelty lies in the automation pipeline, not the interaction paradigm itself. Similarly, Entropulse is a *training schedule* rather than a new RL algorithm; no theoretical or comparative justification is provided beyond empirical observations.

- **Missing broader impact and ethical discussion:** Given that this system trains autonomous agents capable of full computer control, a discussion of potential misuse, safety mechanisms, or privacy implications is notably absent.

### Questions
- Please specify the number of seeds, compute hardware, and total training cost (GPU hours, cluster size). How reproducible are the reported results on smaller scales?
- What are the hardware requirements and cost implications for scaling your distributed RL infrastructure to the reported scale? What performance trade-offs or bottlenecks did you observe in practice?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This work presents ComputerRL, a scalable framework for training autonomous computer-use agents via end-to-end reinforcement learning. ComputerRL introduces an API-GUI paradigm that unifies APIs and GUI actions, enabling higher efficiency and generalization on computer-based tasks. To support large-scale training, the authors develop a distributed RL infrastructure orchestrating thousands of virtual desktops. They also introduce Entropulse, a hybrid training strategy that alternates between RL and supervised fine-tuning, in order to prevent entropy collapse. Applied to GLM-based models, ComputerRL achieves 48.9% success on OSWorld, surpassing prior state-of-the-art agents such as OpenAI CUA o3, Claude 4.0, and gemini 2.5 pro, while demonstrating superior efficiency and stability. This work establishes a robust foundation for scaling RL-based desktop agents.

### Strengths
The paper is overall well written.
- Novel API-GUI paradigm for more generality in computer-based tasks.
- New Entropulse training strategy that mitigates entropy collapse by alternating between supervised learning and RL.
- Scalable and asynchronous RL training pipeline
- Strong empirical results

### Weaknesses
Overall, the contribution lies more in engineering execution than in theoretical advancement.
- Limited algorithmic novelty: primarily builds upon GPRO, and alternating between SFT and RL is similar to exploration-refresh or replay strategies.
- The paper does not fully disentangle how much gain comes from ComputerRL’s methods compared to having strong pre-trained models.
- Limited experimental diversity: they mostly evaluate on OSWorld and OfficeWorld, with no long-horizon or multi-user adaptation or interactions.
- Limited accessibility to reproduction: no analyses on FLOPs or cost of the experiments, and a complex engineering setup.

### Questions
1. Could Entropulse be generalized to other settings, such as text-based reasoning or code synthesis tasks?
2. Why do you think Entropulse's performance plateaued below 50%? What would it take to scale it up?

### Soundness
3

### Presentation
3

### Contribution
3
