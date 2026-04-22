# The State of Reinforcement Finetuning for Transformer-based Agents

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 2, 4, 6, 6

## Abstract
Reinforcement finetuning (RFT) has garnered significant attention in recent years, particularly for enhancing large reasoning models such as OpenAI o1 and Deepseek R1. The appeal of RFT largely stems from its ability to refine model knowledge, better align outputs with user intent, and address challenges associated with limited finetuning data. Despite these advantages, the application of RFT in large Transformer-based generative agents remains relatively underexplored. Although these agents are designed to address multiple tasks through large-scale autoregressive pretraining and share many properties with large reasoning models, current adaptation strategies predominantly rely on supervised finetuning (SFT). In this work, we conduct a systematic investigation of several RFT techniques across a variety of finetuning parameter configurations and meta-reinforcement learning (meta-RL) environments, employing few-shot offline datasets. We provide a comprehensive analysis of RFT algorithm performance under diverse experimental conditions and, based on our empirical findings, introduce a lightweight enhancement to existing RFT methods. This enhancement consistently improves outcomes by combining the strengths of both SFT and RFT. Our findings provide valuable insights for advancing the effectiveness of RFT approaches and broadening their applicability to meta-RL tasks with large Transformer-based generative agents, motivating further research in broader domains.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper “The State of Reinforcement Finetuning for Transformer-based Generative Agents” aims to study how reinforcement finetuning (RFT) techniques, commonly used for aligning large language models such as through RLHF, can be applied to Transformer-based “generative agents.” It positions itself as a comprehensive evaluation of RFT methods in offline reinforcement learning settings, integrating variants such as reward modeling and policy-gradient finetuning. The authors claim that RFT can enhance sample efficiency and reasoning ability in pre-trained Transformer agents and present a benchmark framework comparing different RFT configurations. However, the paper’s objectives remain vague—it is unclear whether it introduces a new algorithm, proposes a benchmark, or provides theoretical insights—and the distinction between “Transformer-based generative agents” and standard LLMs or RL agents is not well defined.

### Strengths
The paper touches on a important topic. The motivation to unify reinforcement-based alignment with multi-task or offline RL adaptation reflects an original attempt to connect two active research directions. The authors make an effort to survey existing RFT approaches and to frame them within a consistent experimental setting, which could, in principle, contribute to clarifying the landscape of RL-based post-training methods. The writing is generally clear at a surface level, and the paper demonstrates awareness of recent developments in RLHF and RFT, attempting to position itself at the intersection of language modeling, reinforcement learning, and meta-adaptation. Overall, while the execution is weak, the underlying idea of systematically analyzing reinforcement-style finetuning for large Transformer-based agents shows conceptual ambition and topical relevance.

### Weaknesses
The main weakness of this paper lies in its lack of conceptual clarity and methodological grounding. It is unclear whether the paper’s goal is to propose a benchmark, a new algorithm, or an analytical study—and this ambiguity undermines its overall contribution. The term “Transformer-based generative agent” is used extensively without a precise definition or boundary relative to standard large language models, prompting confusion about what specific architecture or behavior distinguishes these agents from common RLHF-trained LLMs.

From a technical standpoint, the paper’s use of offline reinforcement learning tasks is poorly justified. RFT and RLHF are inherently online or preference-based alignment methods, not baselines for static offline RL datasets. The decision to evaluate RFT in this context suggests a fundamental misunderstanding of the underlying paradigms. Furthermore, the experimental setup is under-specified: there is no mention of which base model was used, what datasets were employed, or how the finetuning protocols were configured. Without this information, the reported results lack reproducibility and interpretability.

The evaluation design is also inadequate—the authors omit strong baselines such as CQL, IQL, or other modern offline RL methods, and they provide no ablations or sensitivity studies to validate their claims. The comparison across RFT variants remains superficial and largely descriptive, without clear analysis of why certain methods perform differently. Finally, the paper’s writing gives the impression of a survey-like overview rather than a focused, hypothesis-driven study; key references are discussed only at a surface level (e.g., RLHF and GRPO), and theoretical or empirical novelty is minimal.

### Questions
- Could the authors clarify the main objective of the paper — is it intended to serve as a benchmarking effort, a new RFT algorithm, or a conceptual analysis of reinforcement finetuning for Transformer-based agents? Clear positioning would help readers understand how to interpret the experimental results and contributions.

- How do the authors define a “Transformer-based generative agent” in contrast to standard LLMs fine-tuned with RLHF or RFT? Are these agents expected to operate in simulated environments, handle sequential decision-making, or simply produce text-conditioned reasoning? A more precise definition would make the scope of the study much clearer.

- Why were offline RL tasks chosen as the evaluation domain for RFT, which is traditionally an online or human-feedback-based training process? Are there theoretical or practical motivations for believing that RFT can meaningfully improve offline RL policy learning?

- What base model and pretraining setup were used for finetuning? Without information about the backbone architecture, scale, or initial performance, it is difficult to interpret whether the observed improvements come from RFT or model capacity.

- Could the authors provide stronger experimental baselines or ablation studies (e.g., comparing against standard offline RL algorithms such as CQL, IQL, or decision-transformer-based approaches) to substantiate the empirical claims?

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work attempts to provide an earnest and comprehensive comparison view on the state of reinforcement fine-tuning (RFT) compared to supervised fine-tuning (SFT) of RL agents. The authors identify several RFT and SFT methods commonly used in the literature for RFT used for LLMs and apply them to RL agents on different environments (Metaworld, Mujoco) in the few-shot scenario to sparse and dense reward settings. Moreover based on the current observation of methods the authors propose a new method that combines RFT and SFT. Across a wide range of experiments the authors demonstrate the benefits of their newly proposed method, which is usually among the best performing methods. Finally, the authors try to provide useful takeaways beneficial for practitioners.

### Strengths
- The paper attempts to provide an earnest and comprehensive comparison of SFT vs RFT on RL tasks.
- Plenty of experiments for different fine-tuning methods (SFT vs RFT)
- The proposed method seems to generally perform pretty well on the different settings.

### Weaknesses
**Inconsistent results and takeaways**

I like the approach of the authors to try to break down plenty of results into key takeaways, however I found multiple contradictions to the key takeaways, some of which are listed below. This is the main factor why I am currently leaning towards rejection at the moment, if they can be resolved I'd consider increasing my rating.

The paragraph on line 312 states the superiority of QP algorithms and their robustness and broad applicability. Taking a closer look at Table 1, however, the gap to competitors is not large and there is usually a competitor that lies within one standard deviation of QP (e.g. DPO vs QP-SFT for Adapters and Decorators). This means that if a test for statistical significance was performed, the result would be that there is no statistically significant difference. Therefore I recommend to tone down the claiming on superiority and robustness of QP.

In line 368 states that increasing the number of trajectories consistently improves performance for all methods, however looking at e.g. PPO this is not the case. Similarly performance on DPO for expert is sometimes worse with 100 trajectories than with 50 for expert data. Furthermore, Figure 1 is utterly confusing to me, why is there a distinction in dataset quality for different methods? Does that mean the different methods were trained on different fine-tuning data? Or does it just show the same scores as Figure 2, but normalized? If the latter is the case, please correct the caption because at the moment it states "Comparison of finetuning dataset quality" which led to my confusion. Furthermore, if I interpreted it correctly and they essentially show the same information, one of those figures could be moved to appendix to accomodate space for further interpretation/results.

Another example are takeaways in line 372: "Supervised approaches yield stable, imitation-driven performance—particularly
when data quality is mixed", however SFT is one of the worst methods on HalfCheetah (Medium). Yet another example: "(ii) regardless of the algorithmic family, more finetuning trajectories monotonically enhance performance", however CQL on Metaworld (Expert) gets worse with more trajectories.

**Novelty**

One important aspect of the contributions of this work is the investigation of the data quality for fine-tuning, however this has been investigated in prior work already [1]. In particular, this prior work introduces metrics quantifying trajectory quality and state-action coverage and their influence on performance for several offline RL algorithms. It would be helpful to clarify the distinction to this work.

Furthermore, [2] also provides a comprehensive comparison between different parameter-efficient fine-tuning approaches for single-task fine-tuning on Metaworld and DMControl, including a wider variety of PEFT methods than this work. Though they only investigate SFT and no RL fine-tuning. It would be helpful to outline the difference to those works concretely.

[1] A dataset perspective on offline reinforcement learning, Schweighofer et al., CoLLAs 2022

[2] Learning to Modulate pre-trained Models in RL, Schmied et al., NeurIPS 2023

**Conceptual framing**

The authors mention several times throughout the paper that they investigate a metaRL setup, however to me it seems it is more like an actual transfer learning/ multitask fine-tuning setup as in [1]. There are also no conventional metaRL algorithms compared which makes me wonder whether the "metaRL" framing is the correct terminology. Furthermore the terminology of "generative agents" is rather confusing to me, as this terminology is usually employed for LLM-based agentic frameworks [2]. Also the focus on "transformer-based" is not necessary as all the tested algorithms are agnostic to the architecture.

[1] Learning to Modulate pre-trained Models in RL, Schmied et al., NeurIPS 2023

[2] Generative Agents: Interactive Simulacra of Human Behavior, Park et al., UIST 2023



**Choice of methods**

It might be interesting looking into [1] for a LoRA-variant that has been shown to significantly improve upon LoRA based single-task finetuning on the MetaWorld tasks in [2]. This could potentially affect the ranking of the methods in the experiments.

[1] Parameter Efficient Fine-tuning via Explained Variance Adaptation, Paischer et al., ENLSP workshop at NeurIPS 2024

[2] Learning to Modulate pre-trained Models in RL, Schmied et al., NeurIPS 2023

### Questions
- Eq 15: why is the expectation only over (s,a) tuples? why can we not just as in regular double Q-learning use (s,a,s’) tuples for training? 
 - Eq 16: would it make a difference if you sampled multiple actions from the policy and take the max instead of only using $\hat{a}$? I know $\pi_\theta$ already maximizes the q-value, but it might help wich counteracting approximation errors.
 - Line 261 mentions that the fine-tuning dataset is much smaller than the pretraining dataset, it would be helpful to provide actual numbers here. While Table 1 mentions 50 finetuning trajectories, this should be made more explicit in the text and not just in the caption.
 - Table 1: Why is the performance on MetaWorld for Prompt tuning equal across all methods? 
 - Line 320 mentions that PPO and CQL lack inductive priors, what does that mean?
 - Line 323 mentions that switching from Adapters to Decorator for PPO yields 3% improvement, however the average score of Adapters for PPO is larger than the one of Decorator, am I missing something?
 - The final reached reward for the sparse setting (Figure 3) is around the same (sometimes even higher) than for the dense setting, why? What is the delay in the reward? For the same number of update steps I would expect massive differences if there was a substantial delay in the reward as shown in [1], which does not seem to be the case for e.g. PPO or CQL. 
 - Any intuition as to why PPO trained on expert data is better in sparse reward settings than in dense reward settings?
   
 [1] RUDDER: Return Decomposition for Delayed Rewards, Arjona-Medina et al., NeurIPS 2019

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a comprehensive study on Reinforcement Fine-Tuning (RFT) for Transformer-based Generative Agents (TGAs) within meta-reinforcement learning (meta-RL) settings. The authors explore various finetuning algorithms, parameter configurations, and their interplay in adapting TGAs to new tasks. They propose a lightweight enhancement combining Supervised Fine-Tuning (SFT) and RFT, providing empirical evidence of its effectiveness across multiple meta-RL environments. The study’s results show that RFT can improve performance, especially in few-shot and sparse reward settings, and the proposed QP-based methods outperform traditional approaches like SFT.

### Strengths
1.The paper systematically evaluates different RFT methods, providing valuable insights into the trade-offs between various finetuning strategies.

2.The introduction of a lightweight enhancement that combines SFT and RFT is a notable innovation, and the proposed QP-based finetuning methods offer a promising direction for improving model performance in meta-RL settings.

3.The authors conduct extensive experiments across multiple environments (MuJoCo, MetaWorld) with varying dataset qualities and sizes, demonstrating the robustness of their methods.

4.The research addresses a practical challenge in adapting large pre-trained models to real-world tasks with limited data, an important topic given the increasing use of Transformer-based agents.

### Weaknesses
1.While the paper highlights the success of RFT in non-RL domains, the motivation for applying RFT to TGAs in meta-RL environments lacks a strong theoretical justification. The analogy to non-RL models feels speculative, without clearly explaining the structural or optimization similarities that would make RFT effective in RL settings. A more principled explanation would strengthen the rationale.

2.The introduction of QP (Q-guided Policy Optimization) is promising, but it requires more clarification. The explanation of how QP combines RL with SFT is not immediately clear, and the potential advantages and challenges of QP are not adequately addressed. A more concise and focused summary would help readers better understand its benefits.

3.The paper focuses on models with up to 40M parameters but does not address the applicability or efficiency of RFT in larger models, such as those with billions of parameters, which are common in current large language models. A discussion on scalability would provide a more complete picture of the method's practical limitations.

4.The conclusion summarizes the contributions well but could benefit from explicitly discussing the broader implications of the proposed RFT method for meta-RL and real-world applications. Additionally, acknowledging potential limitations, such as the method's dependency on smaller model scales, would offer a more balanced view.

### Questions
1.The paper mentions a “...lightweight improvement DP...” that integrates the advantages of SFT and RFT. However, the term "DP" is not clearly defined.

2.Can the authors provide a more detailed theoretical explanation for why RFT should benefit TGAs in meta-RL, beyond analogy with non-RL models?

3.How does the QP method combine RL and SFT? What are its key advantages and potential challenges in practical applications?

4.Given the increasing size of modern language models, how scalable is the proposed RFT method to models with billions of parameters, and what computational challenges might arise?

5.Could the authors discuss the broader impact of their RFT method on meta-RL and real-world applications, and highlight any potential limitations?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates how to fine-tune a Decision Transformer-style generative agent on a new task with RL in a few-shot way, as existing works mainly investigate supervised fine-tuning of such a generative agent. They compare several existing methods and a newly proposed method called QP which adds an additional term of maximizing the Q-value of the policy to either SFT or DPO. They conduct extensive experimental analysis on Mujoco and MetaWorld to investigate how different fine-tuning algorithm choices and adaptation methods influence the fine-tuning performance, which provide useful insights for future work on RL fine-tuning of generative agents.

### Strengths
1. The paper is clearly motivated with a research gap, i.e., RL fine-tuning of generative agents, to fill. 
2. The paper thoroughly investigate many possible RL ways for fine-tuning, and different ways parameter-efficient fine-tuning methods. 
3. The experiments extensively investigate many problem setting and algorithm choices that may influence the performance of RL fine-tuning.

### Weaknesses
1. My main concern is with the significance of novelty of this paper. It's more like a benchmarking paper instead of proposing a new idea. The QP method proposed by the authors is more like a direct extension of the QT (Hu et al. 2024) algorithm to a multi-task setting, which is limited in originality. 
2. As discussed by the authors, this paper only considers moderate-size models on relatively simple benchmarks like Mujoco locomotion and MetaWorld. Whether the lessons learned from these benchmarking results can be extended to larger-scale models on more realistic tasks or not remains unknown.

### Questions
1. Which checkpoint's performance is reported for each method? The last one after training for a fixed amount of steps or the one with the best evaluation performance?
2. Why not do "real" PPO learning? Is it because you want to do few-shot adaptation?
3. The CQL regularization term in equation 11 seems to be inverse?

### Soundness
3

### Presentation
3

### Contribution
2
