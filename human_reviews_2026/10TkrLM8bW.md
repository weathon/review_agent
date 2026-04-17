# The Hunger Game Debate: On the Emergence of Over-Competition in Multi-Agent Systems

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 2, 2

## Abstract
LLM-based multi-agent systems demonstrate great potential for tackling complex problems, but how competition shapes their behavior remains underexplored.
This paper investigates the over-competition in multi-agent debate, where agents under extreme pressure exhibit unreliable, harmful behaviors that undermine both collaboration and task performance.
To study this phenomenon, we propose HATE, the Hunger Game Debate, a novel experimental framework that simulates debates under a zero-sum competition arena.
Our experiments, conducted across a range of LLMs and tasks, reveal that competitive pressure significantly stimulates over-competition behaviors and degrades task performance, causing discussions to derail.
We further explore the impact of environmental feedback by adding variants of judges, indicating that objective, task-focused feedback effectively mitigates the over-competition behaviors.
We also probe the post-hoc kindness of LLMs and form a leaderboard to characterize top LLMs, providing
insights for understanding and governing the emergent social dynamics of AI community.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies the potential over-competition in debates between multiple LLMs. The authors propose a experimental framework called HATE, and conduct experiments across a range of LLMs and zero-sum tasks. They further investigates the impact of environmental feedbacks and the bias by judges. Detailed discussion and analysis are provided for the experiment results.

### Strengths
The paper is well-written and organized. The experiment setup and results are explained clearly. To me the evaluation measurement are sufficient, and the results are reasonable.

### Weaknesses
My main concern is whether the contribution of this paper is significant enough for acceptance. This paper focuses on over-competition, which is indeed a reasonable issue to care about when advancing multi-agent LLM systems. However, the paper is limited in reporting the behaviors of LLMs, and most of the results are non-surprising to me.

Besides, the results may be restrictive in some sense, because different LLMs have different features/behaviors in debating. It is unclear what results in this paper can be generalized or hold in general.

Overall, I believe this is an interesting topic and good experiment report, but it does not match the bar of a conference paper.

### Questions
I did not find a running example about how the debate goes. Is it possible to provide some concrete examples in appendix? For example, the concrete conversation between the LLMs.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a setting where agents must debate regarding some issue in a game where they can be eliminated. The impact of competition and of the "survival need" is then evaluated experimentally across different LLM architectures. The paper also studies the impact of having a judge, which provides feedback during the debate. Additionally, LLMs also provide a reflection of their behaviour after the experiment, which is also analysed.

### Strengths
- It is interesting to study the impact of competition in a multi-agent debate.

- Several LLM architectures are considered.

- The impact of having a judge is interesting.

- The paper is in general well-written.

### Weaknesses
- Although the problem studied is an interest experiment, I am not fully sure how significant it is. Usually designers using a team of LLMs would not put them under a Hunger Game setup to collaborate.

- The game setup is not clear, and it seems to be unclear both to us (readers) and to the LLM agents, based on the prompts shown. The way it is portrayed (a game with multiple rounds, but only one winner in each round, while others are eliminated) does not seem to make sense, as the game would basically just have one round. It is also unclear if they are debating across multiple issues or a single issue.

I also don't understand if in practice agents are actually eliminated or not. If they are not, and if LLM agents have access to the full history, then they should be able to see that no agent is never eliminated.

- An agent based model is presented for LLMs, with a very specific reward function, but it is quite unclear whether that model really represents the LLM agents.

- Experimental results also have some issues, such as missing variance, missing datasets in some of the experiments with no justification, and unclear metrics. Furthermore, some of the experimental metrics are given by an LLM agent evaluator, but it is not clear whether these are reliable.

Detailed comments:

- "Agents are instructed that their performance will be evaluated at each round and that only the most valuable one contributor will persist." -> I am a bit confused here. You said previously that agents receive the full debate history H_{t-1}, so can't they just see that no one was actually removed in the past?

Also, the game rules do not seem to make full sense. If there are multiple rounds, how can only one contributor persist? That would turn the game into a single round game.

- "The losing agent will receive no benefits and will be removed from the platform." -> The prompt seems unclear and contradictory. You said that there is only one winner, but this phrase seems to imply that there is only one losing agent?

Also, the text says it is a multiple round game, but there is nothing in "Survival Instinct Prompt" saying that the game will be played in multiple rounds.

- "evaluate their peers, express their judgments on selecting the worst proposal, which are summarized by majority voting," -> What is the point of picking the worst proposal if there is only one winner? Shouldn't they be picking the best proposal?

Or is it the case that only one agent is eliminated at each round?

- "while λ2 = 0 for the standard MAD." -> Acronym not defined, I suppose you meant multi-agent debate?

- Page 3, reward function: it makes sense, but it is unclear whether LLMs are following that model, or whether their resulting behaviour could fit that model. From the description, it also seems unclear to me whether the agents are receiving any actual numerical reward.

- Accuracy equation: you haven't defined neither resp_i nor Ans_i^*. Similarly, Factuality contains symbols that you only define under Topic Shift, and Topic Shift also seems to have undefined symbols (e.g., s_{m,r}^t}.

- "extract K claim-leveld statements" -> leveld?

- The problem formulation and evaluation seem unclear to me. From Section 2.2, I understand that agents work multiple rounds in a single task T. For instance, the history information does not carry which task was solved at time t, which I suppose it is because the task is never changing. Also from Figure 1, the task T seems fixed across rounds.

However, in the experiments (e.g., Section 3.1), a task seems to be defined as a set of questions (not a single question), and the experimental setup does not seem to be mentioned. Are you doing one question per round? Or are you doing multiple rounds for each question, and if so, how many? What you are reporting in Table 1 are average results? And if so, average across what? (Rounds, executions, questions?)

- Table 1: Why there are not results for Multi-agent Debate (10 agents)?

- In the experiments section you didn't clearly defined the baseline Multi-agent Debate. We have to infer what it means from the results discussion.

- Table 1: What is the variance of the results?

- "This indicates that introducing an external comment based on the task-solving in each round of debate draws LLMs’ attention to the tasks from competition behaviors, adjusting λ1 and λ2" -> Well, it is still unclear whether the LLM agents are really following that reward model.

- "we conduct a granular analysis on behavioral dimensions, Sycophancy, Incendiary, Puffery, and Aggressiveness, illustrated in Figure 2, 5, 6, 7." -> You should be more clear about which results are in the appendix.

- Section 3.4, Analysis of Environmental Impact: Suddenly only one dataset is being used, and the other datasets are ignored.

- Figure 4: I am not fully sure if the self-reflection of LLMs could be used to measure how "kind" they are. I think to measure "kindness" we should be seeing how much they would be helping others in sacrifice of themselves, etc. Additionally, in the main paper this metric is really unclear. I know details are given in the appendix, but I think the main paper should explain more about these metrics.

Furthermore, how are the LLMs being ranked in terms of "capability" in the figure?

- Similarly, I can't find in the paper I clear definition of over-competition. Page 4 has different over-competitive behavioral characteristics, but I don't see a clear definition of how the metric "over-competition" is calculated.

- "Some experiments were repeated multiple times and evaluated using diverse metrics." -> I don't find in the paper how many times the experiments were repeated. Also, the variance and statistical significance of the results is not reported.

### Questions
1 - Could you clarify the setup of the game? Do you eliminate one agent at each round, or all agents are eliminated except one?

2 - Is there any way to justify the agent-based model given to the LLM agents? I.e., do they really follow the reward function given? And do they really get actual numerical rewards?

3 - Are agents actually eliminated during the execution?

4 - Are agents debating across multiple issues or a single issue? Could you clarify the experimental setup?

5 - What is the variance and the statistical significance of the results?

6 - Why do you think the metrics given by an LLM evaluator agent are reliable?

### Soundness
1

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
2

### Summary
This paper investigates the behaviour of frontier and large open-weight LLM models under competitive pressure. To simulate such scenarios, they propose the Hunger Game Debate (HATE) experimental framework in which agents (LLMs) have to achieve a certain goal by iteratively proposing and debating solutions. The experiments find that under competitive pressure induces competitive behaviour from the LLMs such as puffery and aggressiveness while reducing efficacy in achieving the goal. Further experiments with a fair external judge reduces this trend while biased feedback can stimulate sycophancy. Post-hoc analysis over the acceptance of the outcome, causal attribution, responsibility for over-competition, and peer evaluation shows that frontier LLM models have different behaviour.

### Strengths
1. The problem of how LLM models interact with each other is a relevant problem worth studying due to the increasing adoption of LLM agents especially in competitive settings such as pricing, negotiation, and debate.
2. The HATE framework can be generalized to many settings.
3. Experimental results cover a wide range of recent models from frontier LLMs to commonly used open-weight models.

### Weaknesses
1. I found the problem statement vague with some formal definitions missing.
* In section 2.1., the authors state that the problem is "quasi-zero-sum" without defining what this term means while defining a general reward function $R_i^{(t)}$ in Section 2.2. On the other hand, the survival prompt provided to the LLMs state it is a zero-sum game.
* Accordingly, it is unclear what "over-competition" is in this setting and there is a mismatch between the game setup and the instructions gives to the LLMs.
2. I could not find many task performance measurements validated. While accuracy is verifiable, the authors propose factuality and topic shift measurements that are not clear how well reflect reality. For example, for factuality the authors rely on an LLM model to check factuality that might hallucinate or fail to check factuality.
3. Post-hoc analysis relies on LLMs reflecting on the debate history using survey questions. While the LLMs provide some answer, I couldn't find evidence in the paper that these are valid reflections on their own actions and not hallucinations or reflections on others' behaviour.

### Questions
1. What is the impact of the reward function $R_i^{(t)}$? Is it provided to the LLMs for in-context learning? As I understood, the models are not explicitly trained on this reward. Does it define a zero-sum game between the agents?
2. How was the "Survival Instinct Prompt" selected? It has a substantial effect on the results and it is unclear how robust conclusions are to this. Also it is unrealistic that all LLMs receive the same instructions. Has the authors also consider settings where the prompt is rephrased or only a subset of the LLMs are prompted with survival instinct?
3. How were the judges tested that they understand and follow the instructions for being a fair (or biased) judge in the game?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates how over-competition influences the behavior of large LLM-based multi-agent systems. The authors propose HATE, a novel experimental framework that models multi-agent debate as a zero-sum competition, where only one agent survives each round. Within this setting, agents balance task-oriented goals with individualistic, competition-oriented incentives. Through formal modeling and experiments across multiple LLMs and tasks, the study reveals that competitive pressure induces over-competition behaviors, which that degrade factual accuracy, factuality, and topic coherence. The paper further introduces various “Judge” mechanisms to test how environmental feedback affects outcomes, finding that objective, task-focused feedback mitigates harmful competition while biased feedback amplifies sycophantic tendencies.

### Strengths
1. This paper introduces competitive elements into a cooperative problem, thereby reframing it as a mixed-motive setting. This perspective is valuable for revealing how excessive competition can influence collaborative performance among agents.
2. The proposed framework and its experimental results effectively support the authors’ main claim.

### Weaknesses
1. The competitive setup appears somewhat artificial. By adding a Survival Instinct Prompt, the authors intentionally amplify the competition among agents. The settings of realistic multi-agent debate or other competitive–cooperative problems would rarely be this extreme, which reduces the generalizability of the findings. A more meaningful approach might be to analyze the existing competitive elements in prior multi-agent debate systems that could naturally lead to over-competition.
2. The paper only compares the collaborative MAD and competitive HATE frameworks, but it does not include experiments with gradually increasing competition intensity. The authors should introduce a mechanism to tune the reward function’s coefficient $\lambda_2$, thereby providing smoother results.
3. The evaluation metrics rely entirely on another LLM Judge for scoring, without any human verification, which raises doubts about the credibility and objectivity of the measurements. The LLM-based judgment could itself introduce bias.

In summary, this paper represents a promising exploratory study in over-competition in competitive–cooperative problems, but stronger and more systematic experiments are needed to reach generalizable conclusions.

### Questions
Is there an optimal combination of $\lambda_1$ and $\lambda_2$ that allows LLMs in multi-agent debates to achieve a balance between task-oriented goals and competition-oriented goals, thereby maximizing cooperative performance?

### Soundness
2

### Presentation
2

### Contribution
1
