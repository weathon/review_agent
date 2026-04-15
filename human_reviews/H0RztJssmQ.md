# Adaptive Environmental Modeling for Task-Oriented Language Agents

- Decision: Reject
- Scores: 3, 5, 5

## Abstract
Recent advancements in the realm of intelligent agents, particularly those employing large language models, have been notably significant. Notwithstanding these advancements, intelligent agents encounter substantial challenges, predominantly in interactive and dynamic scenarios such as online shopping, attributed to an absence of integrated environmental modeling. In this paper, we propose a task-oriented environmental adaptation approach, allowing language agents to autonomously model new environments. This approach comprises two pivotal phases: Pre-Task Environment Exploration and In-Task Environment Update. The Pre-Task Environment Exploration phase incorporates a greedy exploration strategy, leveraging an agent in the role of an Evaluator to optimally explore environmental information based on present observations and feasible actions. This strategy is implemented through a recursive algorithm, enabling agents to choose and execute the top-k scored actions, thereby efficiently forming an Action-Observation Tree as the initial environmental modeling. During the In-Task Environment Update phase, agents employ environmental information to enhance task performance. The information generated from task execution and interaction trajectories is used to refine environmental modeling. These processes are iteratively executed, achieving mutual enhancement. We conduct a systematic evaluation of the environmental modeling, assessing both its effectiveness and comprehensiveness. The results demonstrate that under our approach, agents can indeed construct accurate environmental modeling. Simultaneously, we observe a significant enhancement in agent performance on both the ALFWorld-Eco and the WebShop benchmark datasets due to the application of environmental modeling.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors propose a task-oriented environmental adaptation approach to model new environment with two phases: Pre-Task Environment Exploration and In-Task Environment Update. The authors conduct experiments on ALFWorld-Eco and WebShop benchmark datasets to validate the proposed method.

### Strengths
originality

The authors attempt to improve task-oriented language agent with environment model. It would be a novel approach; however, there are issues with the design: both the underlying LLM and the "environment model" from Pre-Task Environment Exploration and In-Task Environment Update are approximations, which will cause problems to agent's performance.

quality

The environment model as the authors build is not the same as defined in standard AI textbooks. The performance metric "score" is not clearly defined.

clarity

There are some issues in the writing as discussed above.

significance

There are issues in the paper as discussed above.

### Weaknesses
Both the underlying LLM and the "environment model" are approximations, which will cause problems to agent's performance.


The performance metric "score" is not clearly defined and the method is not clear and not objective.

### Questions
A.

3.1 PRE-TASK ENVIRONMENT EXPLORATION


"we propose employing the agent itself to evaluate states"
Such self-reference method is fundamentally flawed since the underlying LLM is not perfect, and there is no study about how different this may deviate from a perfect model. 

B.

How to score is not clearly defined.

C.

Algorithm 1
It is for deterministic environments only, i.e., it does not represent uncertainties in action takings / transitions.

D.

When we talk about environment model, we talk about state/observation transition model. Algorithm 1 samples the environment model to build a tree. It is over-claiming to call it modeling the environment.


E.

4.3.1 EVALUATING ACQUIRED ENVIRONMENTAL INFORMATION
Comprehensiveness
"we adopted a manual evaluation method, specifically utilizing a crowdsourcing approach, allowing ten individuals familiar with the relevant environment to score the amount of information contained in the environmental observations corresponding to each action, with a total score of 100"

The method appears very vague, subjective and arbitrary.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper presents a task-oriented environmental adaptation approach for intelligent agents, particularly those using large language models, to autonomously model new environments in dynamic scenarios like online shopping. The approach is divided into two phases: Pre-Task Environment Exploration, which uses a greedy exploration strategy to form an initial environmental model, and In-Task Environment Update, which refines this model based on task execution and interaction data, leading to improved task performance.

I have several major comments and questions:

- In the abstract it’s not clear what is the role of language agents, and ‘language’ in particular in the proposed solution/innovation. All described is the process of recursive greedy exploration and update, a process that is not novel and has been around for some years now.

- Similar issue with the introduction. I still don’t understand what’s the role of language or language agents in the recursive refinement process. The introduction needs to be significantly revised to focus on the central innovation and contribution proposed by this paper, which I believe is around the use of language and language agents, rather than the recursive environment model update, which is not novel.

- It is not very clear to me, from the introduction, what is the critical problem that this approach is trying to solve. The paper needs to be motivated much better. Try providing intuitive examples that why adaptive environment modeling is a significant problem that needs to be addressed. You may also consider adding a mathematically rigorous problem formulation section.

- The first point mentioned in the list of contributions in not a contribution. It more sounds like a motivating question. Please remove. It’s better to only have two contributions than overselling a motivating point as a significant contribution to the field.

- Please apply the revisions suggested in my first two comments to the second contribution item as well. Again, I believe the contribution is around the use of language and language agents for environment modeling, rather than the recursive environment model update. The latter is not novel.

- I read through the methodology, and unfortunately, still don’t have a solid understanding about the role of ‘language’ in the proposed solution. Is language used for the refinement process, for instance by posing critical questions regarding the greedy actions made and their outcome? This is unclear and need to be explained in more details.

- If I understand correctly, during the refinement process, the history of the greedy actions is considered and then the environment model is updated. Is that correct? But what is the update process, specifically, as described in equations 2-3. Also, is there a mechanism to refine the greedy action policy? Is the greedy action policy adopted for the entire model refinement process? How to be sure that the greedy action execution policy is the best approach throughout the model refinement process?

- The evaluations and results are significantly insufficient. More test cases, benchmarks and metrics are needed for a solid conclusion. In the only result figure presented, why are the accuracies so low? Why aren’t any of the ablations learning anything? Such questions need a detailed discussion. Also, error bars must be plotted to see if the differences are statistically significant. This also applies to results and numbers reported in the tables. Overall, the evaluation section is very weak and needs significant improvements.

- What are the limitations of the approach. The limitations are never discussed.

At current states I vote for rejection mostly due to weak motivation, unclear presentation, unclear contributions, and weak evaluation. I need to see more discussions and revisions as suggested above, as well as suggested by my fellow reviewers to make this a ICLR-ready paper. I’d be happy to increase my score when authors satisfactorily addressed the comments.

### Strengths
See above.

### Weaknesses
See above.

### Questions
See above.

### Soundness
3 good

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a novel approach to enable language agents to autonomously model new environments, thereby enhancing their task performance. The approach involves two phases: Pre-Task Environment Exploration and In-Task Environment Update. The authors evaluate the approach using two environments: Webshop and ALFWorld-Eco. The results show that the proposed method is beneficial for modeling environmental to solve task better.

### Strengths
1. The approach enables language agents to autonomously model new environments, thereby enhancing their task performance.
2. The authors evaluate the approach using two environments and provide detailed results.

### Weaknesses
1. Many details in the method were not explained clearly. In the first stage of environmental exploration, the language model is used as a Evaluator for validation. Please provide the corresponding prompt. In the second stage of environment update, no details were provided on how to interactively generate new trajectories and how to update the environment tree using the new trajectories, such as the corresponding prompt.
2. There were also many details that were not clearly written in the experiment. In a Comprehensiveness evaluation, the criteria for manual scoring and the consistency of manual scoring should be given. Is the accuracy used in Table 2 and Figure 2 in ALFWorld Eco. Please provide details on what different stages refer to in Figure 2.
Because many details are not explained clearly, it is also difficult to judge the contribution of the work.

### Questions
N/A

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair
