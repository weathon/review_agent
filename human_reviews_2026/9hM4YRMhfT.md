# GUI-Shepherd: Reliable Process Reward and Verification for Long-Sequence GUI Tasks

- Decision: Reject
- Scores: 4, 2, 2

## Abstract
Autonomous agents for long-sequence Graphical User Interface tasks are hindered by sparse rewards and the intractable credit assignment problem. To address these challenges, we introduce GUI-Shepherd, a Process Reward Model that provides dense, step-by-step feedback to guide agents. GUI-Shepherd is trained on a diverse large-scale data set of 52k interactions that features human-annotated scores and GPT-4o generated rationales, enabling it to serve both as a reward provider for RL training and as a verifier for inference. As far as we know, we are the first to conduct a systematic study of process supervision in GUI agents, across diverse settings from online long-horizon tasks to offline single-step prediction. On the online AndroidWorld benchmark, GUI-Shepherd improves success rate by 7.7 points via multi-turn online PPO, significantly outperforming Outcome Reward Model based competitors. When used as an inference verifier, it brings 5.1 points improvements. The benefits generalize to the offline AndroidControl benchmark, with gains of 2.2 points as a reward provider and 4.3 points as a verifier. Collectively, our results establish that high-fidelity process supervision is critical for building more capable GUI agents and present a generalizable solution.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces GUI-Shepherd, a process reward model (PRM) that offers dense feedback to guide agents in long-horizon Graphical User Interface (GUI) tasks. It marks the first successful application of PRM to online Reinforcement Learning (RL) in this domain. To address the lack of high-quality training data, the authors also present a scalable dual-pipeline approach for creating process supervision datasets and offer systematic validation of PRM in the GUI domain.

### Strengths
1. This paper is the first to apply PRM to online RL in long-horizon GUI tasks and the experimental results show the effectiveness of the proposed model.


2. The proposed data generation pipeline can generate high-quality training data for the GUI Agent tasks.

### Weaknesses
The proposed model still exhibits a significant performance gap compared to state-of-the-art (SOTA) models on the Android World Leaderboard. For instance, "GUI-Owl-7B" achieves a 66.4% Success Rate (SR), while the proposed method only reaches 40.5% SR. Can you explain why this discrepancy occurs? Is the primary reason related to the foundation model, the training algorithm, or the training data?

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes GUI-Shepherd using process reward model (PRM) that delivers step-level supervision and verification for agents performing long-sequence tasks in mobile GUI environments. The PRM is trained on a dataset created by LLM and human annotations, and it is intended to be used in training LLMs during RL training and acting as an action selector during inference time. The trained LLMs with PRM are evaluated in AndroidWorld and offline one-step AndroidControl benchmarks.

### Strengths
-	The experiment covers both online and offline environments, showing improvements in both settings.
-	The hybrid annotation process leverages both human expertise for correctness and LLM-as-judge for efficiency.

### Weaknesses
-	Strong related works are not included as comparison, for example, DigiRL in both online and offline settings.
-	The contribution of applying PRM in training LLM is limited. Despite the equations in Section 4, the contribution does not extend beyond standard RL objective adaptations, and there are no new theoretical analyses of convergence or limitations of PRMs.
-	The temporal and UI diversity mentioned in training data process are unclear. For example, how to ensure data quality (there seems no postprocessing or filtering)? How to measure diversity?
-	It is unclear how the human annotation is performed. How many humans and how to ensure their expertise? What is their annotation instruction? What is the cost? How exactly is the human annotation process performed?
-	The hybrid annotation is confusing. Figure 5 shows the human annotation interface with ‘thought’, which seems to be generated from LLMs. However, the prompt of LLM annotation in Appendix E shows that generating thoughts requires human annotation/ground truth.

### Questions
See the weakness.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper presents GUI-sheppard which is an early instantiation of process-reward models in ui-based tasks. The main contribution is using a reasoning LLM itself as the critic which I found quite interesting. Overall they show many ablations and interesting experiments showing their methodology is sound.

### Strengths
The paper leans more on the empirical contribution in my mind rather than the methodological one (which is good in my books), the idea of using a PRM for web tasks which are innately multi-turn is intuitive. The approach clearly works which is nice I have some problems with the evaluations but will discuss that later. I really like the ablation that discussed training on human-ratings vs GPT-4 ratings, that was interesting, makes me wonder how far off current OS models are from human annotator , since the difference was not very large.

### Weaknesses
While I am largely positive on the general premise of the works and some of the experiments, I have significant issues with the evaluation which prevent me from recommending acceptance at this stage. 

Only one model and one benchmark is evaluated, I would like to see at least two for each. 

No error bars are presented, I understand the cost of running things multiple times but I think this is a necessity where each things should at least report standard errors. 

As far as I can tell no valiantly PPO is reported, RLHF style value function, this is an important baseline to see if the generative process of the PRM is actually really important. 

I would kind of like to see some minimal analysis on the data required to start up the PRM, similar to [1]

Overall i am flexible with my score but I think there are significant things missing for me to raise my score to an acceptance. 

[1] https://arxiv.org/pdf/2507.04103

### Questions
I wonder how other OS models behave as labelers (more so than humans to be honest). 

Is there any qualitative analysis that can be done on the reasoning of the PRM?

### Soundness
3

### Presentation
2

### Contribution
3
