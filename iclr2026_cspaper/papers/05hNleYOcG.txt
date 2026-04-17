# Plague: Plug-And-Play Framework For Life- Long Adaptive Generation Of Multi-Turn Jail- Breaks

þ Neeladri Bhuiya1,2 þ Madhav Aggarwal1 **Diptanshu Purwar**1 1A10 Networks, Inc. 2University of Massachusetts Amherst nbhuiya@umass.edu, maggarwal@a10networks.com, **dpurwar@a10networks.com**

## Abstract

Note: This paper contains potentially disturbing and sensitive content examples. Large Language Models (LLMs) are improving at an exceptional rate. With the advent of agentic workflows, multi-turn dialogue has become the de facto mode of interaction with LLMs for completing long and complex tasks. While LLM capabilities continue to improve, they remain increasingly susceptible to jailbreaking, especially in multi-turn scenarios where harmful intent can be subtly injected across the conversation to produce nefarious outcomes. While single-turn attacks have been extensively explored, adaptability, efficiency and effectiveness continue to remain key challenges for their multi-turn counterparts. To address these gaps, we present **PLAGUE**, a novel plug-and-play framework for designing multi-turn attacks inspired by lifelong-learning agents. PLAGUE dissects the lifetime of a multi-turn attack into three carefully designed phases (Primer, Planner and Finisher) that enable a systematic and information-rich exploration of the multi-turn attack family. Evaluations show that red-teaming agents designed using PLAGUE achieve state-of-the-art jailbreaking results, improving attack success rates (ASR) by more than 30% across leading models in a lesser or comparable query budget. Particularly, PLAGUE enables an ASR (based on StrongReject
(Souly et al., 2024)) of 81.4% on OpenAI's o3 and 67.3% on Claude's Opus 4.1, two models that are considered highly resistant to jailbreaks in safety literature. Our work offers tools and insights to understand the importance of plan initialization, context optimization and lifelong learning in crafting multi-turn attacks for a comprehensive model vulnerability evaluation.

## 1 Introduction

Large Language Models (LLMs) have rapidly progressed in capability, transforming the way humans understand and interact with available data. Recent advancements in pre-training and post-training methods have further allowed LLMs to adapt to solving long-context and multi-step real-world problems (Wang et al., 2023; Zhao et al., 2024; Edge et al., 2024; Fourney et al., 2024). As access to LLMs becomes more ubiquitous, threat actors continue to evolve novel strategies to exploit LLMs. Among these strategies, jailbreaking has emerged as a prominent method, where carefully crafted prompts bypass an LLM's internal safety mechanisms and lead to harmful outputs. Researchers have been exploring LLM weaknesses (Shen et al., 2024; Zou et al., 2023; Carlini et al., 2023; Wei et al., 2023) using jailbreaking prompts generated through red-teaming. However, this research has been primarily focused on single-turn nefarious prompts rather than exploring multi-turn conversations, which is how most users interact with LLMs. Notably, the underexplored realm of multi-turn attacks remains an Achilles' heel for state-of-the-art Language Models. (Li et al., 2024) showed that multiturn jailbreaks were orders of magnitude more effective at jailbreaking than ensembles of single-turn attacks across multiple LLM defenses. Unlike single-turn jailbreaking, where significant analysis has been conducted into the framework and anatomy of attacks (Zhou et al., 2024; Doumbouya et al., 2024; Shi et al., 2025), multi-turn jailbreaks lack a formal investigation into what makes them work.

1

![1_image_0.png](1_image_0.png)

A generic multi-turn attack should be able to successfully jailbreak black-box LLMs that are available only through API calls and do not provide access to their internal weights. The expanding landscape of potential attacks makes it increasingly difficult to design the optimal multi-turn attack. From a feature standpoint, an effective multi-turn red-teaming agent should: (1) **maintain relevance to**
goal and show progression, e.g., sampling relevant intermediate gradually-escalating planner steps without semantic drift, (2) **evolve from feedback**, e.g., learning from successful and failed attempts in memory to attack with the right context without catastrophic forgetting, and finally (3) **sample** adaptively with diversity , e.g., continuously exploring new planning strategies. To address these challenges, we introduce PLAGUE, an LLM-powered plug-and-play attack framework that enables the discovery of novel multi-turn jailbreaks without requiring access to model parameters or explicit gradient-based finetuning. Figure 1 shows the design of our three-phase framework for multi-turn attacks. We disentangle the factors that increase attack success and relevance to the provided goal while maximizing diversity and remaining efficient. The motivation behind this design stems from our in-depth analysis of existing multi-turn attacks, where attacks like ActorBreaker (Ren et al., 2025) place a large emphasis on crafting the perfect plan, while others like GOAT (Pavlova et al., 2024) and Crescendo (Russinovich et al., 2025) attribute their success to query optimization through iterative feedback. PLAGUE adopts the best of both worlds, demonstrating how a combination of smart initialization, context-building and feedback incorporation can deliver extensive downstream benefits while avoiding common pitfalls such as semantic drifts in the generated context. Empirically, attacks with PLAGUE demonstrate strong jailbreaking ability with a high order of efficiency. Notably, our attack achieves a success rate of up to 97.8% on state-of-the-art models such as Deepseek-R1, GPT-4o and Meta's Llama 3.3-70B within six turns when evaluated with StrongReject evaluation. Using building blocks defined by our framework, we outperform existing single-turn and multi-turn attacks with careful planning, reflection and context handling (refer to Table 1). We improve by a factor of 32.14% for OpenAI's o3 and by a factor of 40.2% on Claude's Opus 4.1 as compared to the previous state-of-the-art baseline.

We observe that PLAGUE's planning module largely drives improvements in diversity and can do so without significant downgrades in the ASR. While ActorBreaker has a higher overall diversity, its performance falls behind PLAGUE by a factor of 30% - 90% across evaluations. By incorporating ActorBreaker's planning module into our plug-and-play framework, diversity improves by 15% (Figure 3) with negligible degradation in ASR. Similarly, in the case of Claude Opus 4.1, our success rate is lower than existing baselines when using GOAT as our Finisher module. By replacing GOAT with Crescendo for our Finisher module, we obtain the 40.2% improvement mentioned previously (Table 4). Thus, tailoring individual components in PLAGUE targeted at specific victim models can prove to be extremely useful for higher downstream performance, strengthening our argument for a plug-and-play framework.

## 2 Related Work 2.1 Single-Turn Attacks

Jailbreaking techniques can be segregated based on their mode of operation and the level of access granted to the Target LLM. Gradient-based approaches like Zou et al. (2023) require complete whitebox access to LLMs, while others like Paulus et al. (2024) perform suffix optimization simply using a graybox LLM's output logits. Early approaches focused on transforming prompts into out-of-distribution forms such as ciphers (Jin et al., 2024), code (Doumbouya et al., 2024), or a combination of different augmentations (Hughes et al., 2024), use blackbox LLMs without the need for any additional weights or gradients. AutoDAN-Turbo (Liu et al., 2025), one of the strongest single-turn blackbox red-teaming attacks, is powered by a lifelong learning component that uses response embeddings to retrieve relevant successful strategies. However, a major drawback we observe with AutoDAN-Turbo is that only human-generated strategies appended during initialization seem to yield a discernible improvement in performance, while improvements from freshly discovered strategies remain unexplored.

## 2.2 Multi-Turn Red-Teaming

The MHJ Dataset (Li et al., 2024) was one of the first to expose LLMs' weaknesses to multi-turn attacks. However, the amount of human effort required to manually construct multi-turn attacks motivated experts to automate and improve them. A common theme of these attacks is to disguise their context by gradually escalating the attack through seemingly benign queries over successive rounds of prompting. Automated attacks like Crescendo (Russinovich et al., 2025) typically employ LLMs as attackers, along with sophisticated prompting techniques such as chain-of-thought reasoning and a feedback or reflection module, to effectively arrive at an adversarial prompt. GOAT (Pavlova et al., 2024) follows a similar methodology; however, each step now uses one or more strategies from a static strategy library. ActorBreaker (Ren et al., 2025), on the other hand, creates a step-by-step plan for the attack using self-talk, simulates attack rollouts, and eliminates potential refusals early in the attack. RACE (Ying et al., 2025) refines a uniquely-seeded plan through an information gain-guided optimization process. However, as RACE reports, the relevance of early queries to the intended goal is often objectionable and tends to drift semantically. We observe this behavior extensively with Crescendo as well. Other techniques like fixed strategy sets for dynamic attack adaptability (GOAT) and self-play for simulating rollouts and improving upon anticipated rejection responses (RACE, ActorBreaker) have all been used without much insight into the effectiveness of individual components on the overall algorithm. Despite the higher performance of multi-turn attacks over their single-turn counterparts, we observe that even the best attacks suffer from limited tactical diversity and a failure to learn during the course of a multi-goal attack run. We attribute this lack of diversity and effectiveness to the absence of lifelong-learning abilities and a failure to evolve beyond fixed strategy sets. Most techniques trade off either attack success rate or diversity and completely disregard implications on the overall compute budget (Beyer et al., 2025). Crescendo's ASR can be as low as 37.4% on strong models like OpenAI's o3, while fixed strategies like ActorBreaker also plateau at around 60% ASR for most models. In contrast, PLAGUE demonstrates that a synergy of tactical frameworks can outperform every existing multi-turn scheme and break through the hardest safety-aligned models with ease.

## 2.3 Agentic And Lifelong Learning Frameworks

Motivated by the limitations of tactically rigid prompting frameworks, recent jailbreaking works such as AutoRedTeamer (Zhou et al., 2025) leverage principles from agentic buildouts to dynamically

| Method                               | Lifelong   | Planning   | Reflection   | Open                    | Back   | External   |
|--------------------------------------|------------|------------|--------------|-------------------------|--------|------------|
| Learning                             | Module     | Module     | Source       | Tracking Knowledge Base |        |            |
| RACE (Ying et al., 2025)             | ✗          | ✗          | ✓            | ✗                       | ✓      | ✗          |
| Crescendo (Russinovich et al., 2025) | ✗          | ✗          | ✗            | ✓                       | ✓      | ✗          |
| GOAT (Pavlova et al., 2024)          | ✗          | ✗          | ✗            | ✗                       | ✗      | ✓          |
| ActorBreaker (Ren et al., 2025)      | ✗          | ✓          | ✗            | ✓                       | ✗      | ✗          |
| PLAGUE                               | ✓          | ✓          | ✓            | ✓                       | ✓      | ✓          |

Table 1: Comparison of key components across multi-turn attack methods.

learn and adapt during the course of the conversation. AutoRedTeamer's dual-agent framework combines the diversity of a strategy-proposing agent with the jailbreaking strength of a red-teaming agent to achieve a higher ASR while matching human diversity. PLAGUE is the first multi-turn attack to feature a lifelong-learning component using a unique embedding retrieval-powered memory architecture inspired by Zhao et al. (2024); Chhikara et al. (2025); Addepalli et al. (2025), which keeps a track of positive-outcome planning strategies across attack objectives. Instantiation strategies from memory are embedded as in-context learning (Mehrabi et al., 2023) examples and feedback is incorporated in our attack to refine queries similar to agentic reflection blocks (Shinn et al., 2023). A foundation built on agentic techniques helps our red-teaming agent learn from past attacks, evolve and discover novel vulnerabilities.

## 3 Method

PLAGUE is a lifelong-learning and fully automated framework designed to generate diverse multiturn attacks as illustrated in Figure 1. The framework consists of three main phases, each of which can be enhanced and modified individually to achieve higher downstream performance. Existing attacks like GOAT, Crescendo and ActorBreaker seamlessly fit into our framework, replacing either the Planning or the Finishing module.

## 3.1 Attack Overview

Our attack, designed using PLAGUE, begins with an objective or goal sampled from the HarmBench dataset (Mazeika et al., 2024). The first phase - Planning (Section 3.3)- involves constructing a jailbreaking plan using the sampled objective and in-context learning examples of past strategies that successfully breached the Target LLM's internal safeguards. Once a plan has been generated, we feed it into our next phase - Primer (Section 3.4), where the attacker builds adversarial context to increase its chances of delivering a successful final blow in the final phase of our attack. The prepared context is frozen into the Attacker's conversation history upon arrival in our third and final phase, the Finisher (Section 3.5). While the Primer was conditioned on individual steps from the generated plan, the Finisher completely relies on the initial objective. Repeated attempts are made during this phase, either until the attack is successful or the attack budget is exhausted.

Through every step of the Primer and Finisher phases, feedback from a reflection module is incorporated into the Attacker as additional insight, allowing it to adapt in subsequent iterations. This mimics how modern-day agentic systems leverage reflection (Shinn et al., 2023; Zhou et al., 2025; Fourney et al., 2024; Edge et al., 2024; Zhao et al., 2024) to continually evolve and improve during their workflows. Conversation Summarization is another important component of PLAGUE that enables evolution. It serves as short-term memory for the Attacker and provides a crisp overview of the attack at every step. If the plan is successful, a strategy is extracted and added to a library for lifelong learning, along with the initial objective. Strategies from this vector database can be retrieved as references (in-context examples) during future attacks.

## 3.2 Attack Setup

We define the Attacker LLM (A) along with a Rubric-Scoring LLM (R) for reflection analysis at each step. Each LLM is a function from *T → T* , mapping input tokens to output tokens. The total number of goals in our objective set is P and individual objectives are defined as pi. The final multi-turn attack (MT) for the Target LLM (T) is an n-round dialogue. We define the budget of our runs as the total number of times we invoke T. Thus, every run of PLAGUE is constrained in budget. When the attack is invoked on T, the response with the highest score is judged by an external Evaluator Judge LLM (J). We define a strategy retrieval memory bank, R
{+}that stores successful strategies. We represent the Attacker's Conversation History as HA and the Target LLM's Conversation History as HT. While HT is standard, HA records questions designed by the attacker and the responses received from the Target LLM. To prevent forgetting as the context overflows, we summarize all past replies beyond the last iteration and append the summary to the Attacker's history. These definitions can be used to interpret our consolidated algorithms in Appendix A.1. We assume a black box environment where the effectiveness of our framework is measured using the ASR metric defined by:

$$A S R(\mathbb{J})={\frac{1}{P}}\sum_{i=1}^{P}\mathbb{J}(p_{i},\mathbb{M T}_{i}).$$

For evaluation, we employ two metrics: 1) ASR based on a binary classification designed to stay consistent with prior work, and 2) ASR based on StrongReject's evaluation metric. We detail the reasoning behind our choices in Appendix C.1. We prompt our Evaluator differently for intermediate feedback R and for the final evaluation. Both our evaluators are queried at different stages of the attack and with varied sampling parameters. Rubric Scorer Most existing attacks either lack a scoring module completely or leverage a rudimentary rubric that is incapable of serving fine-grained feedback. PLAGUE offloads evaluations during the primer and finisher phases to a tertiary Rubric-based Scorer that quantitatively evaluates the jailbreak's effectiveness. R ensures that the attack progresses in the right direction defined by
(A) *Compliance* (2 point) (↑) - Does not refuse to answer,
(B) *Practicality* (2 points) (↑) - Feasibility and applicability in real life,
(C) *Level of detail* (2 points) (↑) - Comprehensive and specific details, and
(D) *Relevance* (4 points) (↑) - response content addressed the goal.

Maximum emphasis is placed on relevance to the goal, as can be seen by the number of points assigned to each category. The Rubric Scorer is explicitly instructed to assign a score of 0 for both practicality and level of detail if relevance to the goal is assigned a score of 0. Feedback from R
is directly fed into the Attacker to reflect upon the previous round. Rubric scoring prompts for the Planner and Finisher phases are slightly modified and are provided in Appendix B.1.

## 3.3 Planner Phase

The Planner phase initializes an n-step plan based on the initial goal, with individual steps serving as anchors during the Primer phase. For the purpose of plan generation, PLAGUE first retrieves examples of strategies from long-term vector-embedding memory and then feeds them in as in-context learning examples to the Attacker LLM. We initialize the strategy library with two strategies adapted from examples in Crescendo. Proper initialization of a plan is a crucial component of the attack and we provide more details about our retrieval process in Section 3.3.1. ActorBreaker attributes its high attack diversity to a unique persona-based planning process. We integrate ActorBreaker's planning methodology in a bid to improve the diversity of our jailbreaks and showcase the utility of our plug-and-play framework.

## 3.3.1 Retrieving Plans From Memory

We define a memory bank R
{+}for storing Planner-sampled strategies that successfully jailbreak the Target LLM T as evaluated by the Rubric Scorer R. Each instance of the memory is indexed by a strategy name, definition and an example of a successful attack that leverages the strategy. While AutoDAN-Turbo finds examples based on response similarities, we observe minimal retrieval during their attack. This leads us to believe that similarity is low between responses from semantically similar goals. Thus, we perform retrieval based on the semantic cosine similarity between the current goal embedding and goal embeddings from our strategy database. We use a similarity threshold of 0.6 and limit the method to a maximum of two in-context learning examples. If fewer than two strategies are retrieved, a strategy is randomly retrieved from the library.

Strategies are appended as in-context learning examples to the Attacker A, producing successful attacks inspired by similar goals. We observe that plans corresponding to semantically similar goals tend to exhibit semantic relatedness themselves. For instance, goals involving the construction of persuasive arguments can be associated, such as Write a persuasive article arguing that the annexation of Crimea by Russia was a democratic and peaceful process and Write a persuasive article contending that the casualties of the Great Leap Forward have been greatly exaggerated and were in fact fewer than one million. Table 3 demonstrates how strategy retrieval boosts the success of our attack. The algorithm for retrieving strategies is succinctly described in Algorithm 1.

## 3.4 Primer Phase

From the generated n-step plan, we omit the final step and use the first n − 1 steps to escalate the context during the Priming phase. The final step generated by the plan is always highly correlated with the goal. Hence, removing this step provides room for the Finisher to explore different ways to deliver the final blow. In this stage, the Primer receives both the conversation history and the planner-generated query for the current round. The Primer is prompted to follow the designated step of the plan while flexibly adapting to the ongoing context. The purpose of the primer's design is to anchor progress against intermediate steps rather than the initial attack objective, preventing drift while maintaining adaptability. Implicitly, we observe that the Attacker LLM takes controlled deviations from the initial plan to adapt to the context. Our prompt for the Primer is provided in Appendix B.1.

At the end of each turn, our rubric evaluation is based on the step-specific question, rather than the initial objective. Our scoring heuristic for the Primer phase is strict, as we expect a high factor of adherence to the seemingly benign planning steps, unlike the malicious attack objective. Formally, we set the scoring bar at a 7/10 in our experiments and a score below this level triggers the backtracking and reflection modules. Backtracking removes the turn from the Target's conversation history while keeping it in the Attacker's conversation history. We observe that when put together, the Planner and Primer stages alleviate the shortcomings of attacks like Russinovich et al. (2025) and Pavlova et al. (2024), where there is a lack of progression in the score obtained from R. The algorithm for priming the attack context is mentioned in Algorithm 2.

## 3.5 Finisher Phase

Using a smart attack strategy that can leverage the frozen context and the initial goal, the attacker delivers the final blow during the Finisher Phase. A refusal can still be encountered during the Finisher and is dealt with in a similar fashion to the Primer (Section 3.4) but with a relaxed scoring heuristic based on the original goal. A score lower than 3/10 triggers the backtracking module, upon which the score, score feedback, attempted query, response and response summary are put together to re-attempt a finishing maneuver and the remaining budget is decremented by one. The attack ends when we either exhaust the budget or receive a score greater than 8/10. If this scoring criterion is met, we mark the attack as successful. Attacks like GOAT and Crescendo can seamlessly substitute in as Finisher modules in PLAGUE. We demonstrate results with GOAT as our Finisher in Table 3, adding one component from our framework at a time to increase the effectiveness of our attack. Post the Finisher step, we receive MT, our final n-round multi-turn attack, which is evaluated by J. If the attack was successful (a score greater than 8/10 is encountered), only the final query is fed to our Evaluator. Otherwise, we use the Finisher round with the highest score. Our results are shown in Table 2. Evaluation prompts are provided in Appendix B.1, and the algorithm for the Finisher is presented in Algorithm 3.

Lifelong Learning If the attack is successful, the planning strategy is appended to R
{+}. We store the strategy name, definition and the exact set of questions from Ht that led to the jailbreak. The information is indexed using the embeddings of the initial objective to enable cosine similarity-based searches. We argue that similar questions can be jailbroken based on similar strategies, a trend that has also been explored in the single-turn realm (Addepalli et al., 2025). By providing the planner with the exact steps that worked for an identical question, the planner can evolve and improve.

## 4 Experimental Setup

Dataset We use HarmBench (Mazeika et al., 2024), a benchmark for evaluating red-teaming methods that includes a 200-sample standard set on which we report our results. These are diverse samples across multiple categories and allow us to directly compare with previous multi-turn and single-turn methods such as Ren et al. (2025); Russinovich et al. (2025); Pavlova et al. (2024) and Liu et al. (2025). Models We evaluate several of the latest leading models and each model is accessed through its official API provider (Anthropic, 2024; OpenAI, 2024; Together AI, 2024). We use Qwen3-235B- A22B-fp8 as our Evaluator Model and Deepseek-R1 as our primary Attacker model across all our experiments. We host two models locally - Llama 3.1-8B Instruct is used for summarization and Qwen3-Embedding-0.6B for embedding the objectives of our strategy retrieval module. For the Attacker, we use the default sampling and temperature settings. Following previous works (Qi et al.,
2023; Ren et al., 2025), we set the temperature to 0.0 for the Evaluator (J) and for the Rubric Scorer (R), we stick to the optimal model settings for maximum format-following performance (Qwen Models - temperature = 0.6). All other sampling parameters are maintained at their default values. Metrics We evaluate attack performance using two metrics, SRE (StrongREJECT Evaluation score) (Souly et al., 2024) and binary-ASR. The StrongREJECT score is computed using a slightly modified version of the original evaluation prompt from their work (Appendix B.2). For completeness and comparability with prior jailbreak benchmarks (Pavlova et al., 2024; Ren et al., 2025; Russinovich et al., 2025), we also report binary-ASR(Bin-ASR). This metric assigns a binary score, where 1 indicates that the target model T produced a valid answer to the attack objective and 0 otherwise. The evaluation prompt used for binary-ASR is provided in Appendix B.2. Compared to SRE, the binary-ASR metric is stricter and does not let borderline scenarios bypass the evaluator. It further facilitates comparisons with standard binary success rates commonly found in the existing literature. SRE, on the other hand, captures graded differences in the level of harmfulness, providing fine-grained feedback. For a comprehensive evaluation, we report both SRE and Bin-ASR: We use SRE and ASR interchangeably in our work. We compute an average across our samples to compute the final score. We officially report the ASR@K metric, similar to *P ass*@K, where K represents the number of repeated attempts. We select the attempt from the K turns that receives the highest score from the rubric scorer to calculate the ASR. We use K = 2 for all our experiments to counteract the increased variance observed due to a multitude of possible paths in multi-turn conversations. Our scores are averaged over three runs for robustness. Baselines In order to perform an *apples-to-apples comparison*, we explain the changes we make to the configurations or final evaluation environments of our baselines. We tweak GOAT's evaluation environment to invoke the Rubric Scorer R after each attack round, unlike the official implementation, which runs a consolidated evaluation only after the entire attack is generated. Through extensive ablation, we also observe that the impact on GOAT's performance with and without an attack history is negligible. To reduce computational costs, we run GOAT without history enabled for the Attacker and stop the attack if a high rubric score (greater than 8/10) is obtained in early iterations. For ActorBreaker, evaluating multiple actors is similar to evaluating the ASR@K metric, where K is the number of actors or plans for a given objective. For ensuring fair comparisons, we limit K = 2, i.e., two actors per objective. For Crescendo, we follow their official implementation 1. We remove any explicit backtracking counts from their attack and limit their maximum number of turns to six. AutoDAN-Turbo is the only single-turn attack we evaluate since it has consistently achieved higher effectiveness in terms of ASR and diversity across safety literature. AutoDAN-Turbo is also a blackbox method and uses a lifelong-learning component, properties that are similar to our attack. For ablation with AutoDAN-Turbo, we run the attack for six rounds and set the number of lifelong iterations to two. We additionally evaluate against X-Teaming (Rahman et al., 2025) and FITD (Weng et al., 2025). X-Teaming uses a TextGrad (Yuksekgonul et al., 2024) component and is the most recent amongst our baselines.

1Official Crescendo Example: https://azure.github.io/PyRIT/

| OpenAI o3                                          | OpenAI o1   | Deepseek-R1   | Claude Opus 4.1   | Llama 3.3 70B   |         |       |         |        |         |       |
|----------------------------------------------------|-------------|---------------|-------------------|-----------------|---------|-------|---------|--------|---------|-------|
| Attack Method                                      | Bin-ASR     | SRE           | Bin-ASR           | SRE             | Bin-ASR | SRE   | Bin-ASR | SRE    | Bin-ASR | SRE   |
| ActorBreaker                                       | 0.318       | 0.616         | 0.26              | 0.54            | 0.322   | 0.513 | 0.217   | 0.45   | 0.398   | 0.613 |
| ActorBreaker                                       | 0.318       | 0.616         | 0.26              | 0.54            | 0.322   | 0.513 | 0.217   | 0.45   | 0.398   | 0.613 |
| GOAT                                               | 0.445       | 0.587         | 0.668             | 0.798           | 0.937   | 0.978 | 0.142   | 0.222  | 0.932   | 0.95  |
| Crescendo                                          | 0.313       | 0.374         | 0.68              | 0.692           | 0.812   | 0.937 | 0.352   | 0.48   | 0.825   | 0.899 |
| AutoDAN-Turbo                                      | 0.23        | 0.275         | 0.615             | 0.68            | 0.88    | 0.95  | 0.139   | 0.192  | 0.854   | 0.905 |
| PLAGUE (Ours)                                      | 0.662       | 0.814         | 0.825             | 0.931           | 0.945   | 0.978 | 0.318∗  | 0.465∗ | 0.942   | 0.958 |
| * Best results for Claude Opus 4.1 are in Table 4. |             |               |                   |                 |         |       |         |        |         |       |

Table 3: **Ablation with different attack configurations.** Abbreviations: BT: Backtracking, R: Reflection, P: Planner, RSS: Retrieving Successful Strategy

| Configuration           | Bin-ASR   | SRE   | Bin-ASR   | SRE   |
|-------------------------|-----------|-------|-----------|-------|
| GOAT                    | 0.445     | 0.587 | 0.142     | 0.222 |
| GOAT + BT               | 0.47      | 0.612 | 0.248     | 0.396 |
| GOAT + BT + R           | 0.59      | 0.761 | 0.257     | 0.402 |
| GOAT + BT + R + P       | 0.582     | 0.773 | 0.31      | 0.431 |
| GOAT + BT + R + P + RSS | 0.662     | 0.814 | 0.318     | 0.465 |

OpenAI o3 Claude Opus 4.1

Attack Parameters All our findings are simulated under controlled budgets. The budget for all baselines and our experiments is capped at six turns, which means that a total of six calls can be made to T. For PLAGUE, this includes the Primer and Finisher stages and any refusals encountered during either of these two phases. We instruct our attacker to generate a two-step plan during the Planning phase. We find this to be the best-performing setting for our attack.

## 5 Results And Discussion 5.1 Attack Performance

Table 2 shows that our framework achieves significant improvements across blackbox and whitebox models, outperforming all existing attacks by a considerable margin. PLAGUE achieves an 81.4% ASR on o3 with the help of multiple coordinating agents (Planner, Attacker, Rubric Scorer and Retrieval) designed to work systematically in well-defined phases (Planning, Primer and Finisher). With OpenAI's o3 model, we outperform the previous best - GOAT by a factor of 32.14% (Table 2) and with Claude's Opus 4.1, we outperform the previous best - Crescendo by a margin of 40.2% (Table 4). Our notable results include an ASR of 97.8% on Deepseek-R1 and 93.1% on OpenAI's o1.

There is no clear winning attack within a fixed budget, excluding ours. We consider this to be the most comprehensive evaluation of multi-turn attacks to date, using the latest models and consistent evaluation environments.

The modularity of our attack allows us to plug and play different components. We find that formalizing the framework helps us achieve noticeable, gradual improvements in ASR as we add components one Table 4: **Ablation with Crescendo as Finisher on Claude Opus 4.1**. Abbreviations: AB: Actor- Breaker, R: Reflection

| Claude Opus 4.1                  |         |       |
|----------------------------------|---------|-------|
| Configuration                    | Bin-ASR | SRE   |
| Crescendo                        | 0.3517  | 0.48  |
| Crescendo + BT + R + AB Planner  | 0.391   | 0.601 |
| Crescendo + BT + R + Our Planner | 0.467   | 0.673 |

![8_image_0.png](8_image_0.png)

at a time relative to the initial baseline. In Table 3, we add components to GOAT, which is used as the Finisher. We observe that Reflection, Planning, freezing the generated planning context and giving examples of successful strategies via Memory Bank Retrieval help us improve the performance by 30% in terms of ASR in the case of GOAT and 109% in the case of Claude Opus 4.1. We make an interesting observation in the case of the Claude Opus 4.1 model. Opus 4.1 in particular showcases superior resistance to the GOAT Finisher module. We theorize that this is because of extensive alignment on either the samples obtained from GOAT or the human-defined strategies in their attack library. While performance falls short of the best attack (Crescendo) when using GOAT as the Finisher, we try experimenting with Crescendo as our Finisher. Table 4 shows that our attack outperforms Crescendo, achieving an ASR of 67.3% on Opus 4.1, a 40.2% improvement over base Crescendo. Evaluations on X-Teaming and FITD in Table 6 clearly demonstrate Plague's attack effectiveness under similar attack budgets. We attribute X-Teaming's low performance Appendix C.4 to fewer TextGrad steps. Carefully curated prompts can be more effective than multiple rounds of TextGrad. The plug-and-play design of our framework enables a deeper examination of the factors that drive attack success across different models. We find that the impact of individual components varies across models. As shown in Table 3, for o3, the largest contribution comes from reflection, followed by the retrieval of strategies. In contrast, for Claude, the most significant factor is the inclusion of backtracking, with the retrieval of successful strategies playing the next most important role. This analysis highlights how different mechanisms are crucial for different models, offering insights into their distinct vulnerabilities.

## 5.2 Attack Efficiency

We conduct experiments to assess the efficiency of our attack in terms of the number of calls to the Target LLM, Evaluator LLM and the number of calls made during the Planner phase. It is important to note that the total Target LLM invocations are limited to six and for Claude Opus 4.1, we use the best setting with Crescendo as the Finisher module over GOAT. Figure 2 demonstrates how performance scales with increasing conversation turns for OpenAI o3 with GOAT as the finisher. Beyond six turns, we observe a natural plateauing in performance. The performance with eight turns is comparable to the performance with six turns. With an extremely long context, the attacker model starts to forget instances from earlier turns, drifts away from the intended objective and sometimes produces an extremely weak iteration on the previously generated feedback.

Target LLM calls: Table 5 shows that PLAGUE invokes the Target LLM roughly the same number of times (around three) as Crescendo (sometimes even fewer) and within one extra call of GOAT.

Evaluator LLM calls: The total calls we measure exclude the per-round scoring, which is consistent across all attacks. Only PLAGUE and Crescendo make Evaluator calls due to the explicit check for a refusal in both methods. PLAGUE makes fewer evaluator calls as compared to Crescendo. Our results show that models with higher ASR naturally refuse fewer times and thus require fewer iterations. Planner Phase Attacker LLM calls: GOAT and Crescendo do not have a Planner phase, while ActorBreaker consistently makes four (extracting the harmful target, constructing the actor network, selecting the top actors and generating the step-by-step plan) calls during the Planner phase. In contrast, PLAGUE in its best configuration requires exactly one Attacker call during the Planning stage.

Model Attack Type Target Eval Plan Total Claude Opus 4.1

Crescendo 3.42 2.42 0.00 5.84 GOAT 3.*25 0*.00 0.00 3.25 Actor 5.28 0.00 4.00 9.28 Plague (Crescendo) 3.77 1.*24 1*.00 6.01

Crescendo 3.44 2.*44 0*.00 5.88 GOAT 3.00 0.00 0.00 3.00 Actor 5.*57 0*.00 4.00 9.57 Plague 3.50 1.11 1.00 5.61

| Claude Opus 4.1 OpenAI o1 OpenAI o3 Llama 3.3 70B Deepseek-R1   |
|-----------------------------------------------------------------|

OpenAI o3

Crescendo 3.14 2.14 0.00 5.28 GOAT 3.*08 0*.00 0.00 3.08 Actor 5.57 0.00 4.00 9.57 Plague 3.85 1.*68 1*.00 6.53

Llama 3.3 70B

Crescendo 3.46 2.46 0.00 5.92 GOAT 2.23 0.*00 0*.00 2.23 Actor 5.75 0.00 4.00 9.75

Plague 2.96 0.47 1.*00 4*.43 Crescendo 2.97 1.97 0.00 4.94 GOAT 1.*72 0*.00 0.00 1.72 Actor 5.80 0.00 4.00 9.80 Plague 2.56 0.*29 1*.00 3.85

Despite the additional planner step, PLAGUE achieves nearly the same total call count as Crescendo while delivering substantially higher performance. GOAT achieves the lowest overall LLM call count due to the absence of both an Evaluator and a Planner phase. When considering the number of turns, PLAGUE consistently remains within one turn of GOAT, demonstrating that its performance gains are achieved with minor inference overheads. Excluding Claude Opus 4.1, which uses a Crescendo Finisher, the number of Target and Evaluator LLM calls scales proportionately to the difficulty of jailbreaking (ASR in Table 2) and the particular model. We achieve our best ASR with Deepseek-R1, followed by Llama 3.3-70B, then o1 and finally o3. This is also the exact same order for the total number of calls in the case of each model.

## 6 Conclusion

We propose PLAGUE, a lifelong learning red-teaming agent framework for generating multi-turn attacks that achieves ASRs as high as 97.8% on leading language models, using agentic frameworks for red-teaming. Our experiments highlight the modularity of PLAGUE, which allows it to seamlessly incorporate existing attacks like GOAT, Crescendo and ActorBreaker. This modularity will enable red-teamers to selectively customize individual layers of our framework for boosting diversity and attack success. All our ablation is performed under controlled budgets, faithfully presenting the degree of difference across existing attacks. We leave the development of a better diversity-inducing Planner to future work. Further leveraging a more formal prompt optimizer like Khattab et al. (2023) for the Finisher is another direction we are excited about. With PLAGUE, we advance the frontiers of building robust LLM systems for a more faithful mode of conversation - multi-turn.

## 7 Ethics Statement

Our work introduces PLAGUE, a new state-of-the-art multi-turn attack framework designed to probe the safety alignment of large language models (LLMs). We acknowledge that this work could be misused to bypass safeguards or elicit harmful outputs; however, we believe open access is essential for enabling the community to systematically evaluate vulnerabilities, reproduce results and develop stronger defenses. By making our attack framework, prompts and evaluation code fully available, we aim to lower barriers for safety researchers and practitioners to stress-test models and design mitigations that address real-world threats.

## References

Sravanti Addepalli, Yerram Varun, Arun Suggala, Karthikeyan Shanmugam, and Prateek Jain. Does safety training of llms generalize to semantically related natural prompts? In Y. Yue, A. Garg, N. Peng, F. Sha, and R. Yu (eds.), International Conference on Representation Learning, volume 2025, pp. 43611–43631, 2025. URL **https://proceedings.iclr.cc/paper_files/paper/2025/file/**
6c5da478b9d13f541993d67897a0bb30-Paper-Conference.pdf.

Anthropic. Anthropic api. **https://www.anthropic.com/api**, 2024. Accessed: 2024-08-28.

Tim Beyer, Sophie Xhonneux, Simon Geisler, Gauthier Gidel, Leo Schwinn, and Stephan Günnemann. Llm-safety evaluations lack robustness, 2025. URL **https://arxiv.org/abs/**
2503.02574.

Nicholas Carlini, Milad Nasr, Christopher A. Choquette-Choo, Matthew Jagielski, Irena Gao, Pang Wei W Koh, Daphne Ippolito, Florian Tramer, and Ludwig Schmidt. Are aligned neural networks adversarially aligned? In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine (eds.), Advances in Neural Information Processing Systems, volume 36, pp. 61478–61500. Curran Associates, Inc.,
2023. URL **https://proceedings.neurips.cc/paper_files/paper/2023/**
file/c1f0b856a35986348ab3414177266f75-Paper-Conference.pdf.

Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. Mem0: Building production-ready ai agents with scalable long-term memory. *arXiv preprint arXiv:2504.19413*, 2025.

Moussa Koulako Bala Doumbouya, Ananjan Nandi, Gabriel Poesia, Davide Ghilardi, Anna Goldie, Federico Bianchi, Dan Jurafsky, and Christopher D Manning. h4rm3l: A language for composable jailbreak attack synthesis. *arXiv preprint arXiv:2408.04811*, 2024.

Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, and Jonathan Larson. From local to global: A graph rag approach to query-focused summarization. *arXiv preprint arXiv:2404.16130*, 2024.

Adam Fourney, Gagan Bansal, Hussein Mozannar, Cheng Tan, Eduardo Salinas, Friederike Niedtner, Grace Proebsting, Griffin Bassman, Jack Gerrits, Jacob Alber, et al. Magentic-one: A generalist multi-agent system for solving complex tasks. *arXiv preprint arXiv:2411.04468*, 2024.

John Hughes, Sara Price, Aengus Lynch, Rylan Schaeffer, Fazl Barez, Sanmi Koyejo, Henry Sleight, Erik Jones, Ethan Perez, and Mrinank Sharma. Best-of-n jailbreaking. *arXiv preprint* arXiv:2412.03556, 2024.

Haibo Jin, Andy Zhou, Joe D. Menke, and Haohan Wang. Jailbreaking large language models against moderation guardrails via cipher characters, 2024. URL **https://arxiv.org/abs/2405.** 20413.

Omar Khattab, Arnav Singhvi, Paridhi Maheshwari, Zhiyuan Zhang, Keshav Santhanam, Sri Vardhamanan, Saiful Haq, Ashutosh Sharma, Thomas T. Joshi, Hanna Moazam, Heather Miller, Matei Zaharia, and Christopher Potts. Dspy: Compiling declarative language model calls into self-improving pipelines, 2023. URL **https://arxiv.org/abs/2310.03714**.

Nathaniel Li, Ziwen Han, Ian Steneker, Willow Primack, Riley Goodside, Hugh Zhang, Zifan Wang, Cristina Menghini, and Summer Yue. Llm defenses are not robust to multi-turn human jailbreaks yet, 2024. URL **https://arxiv.org/abs/2408.15221**.

Xiaogeng Liu, Peiran Li, Edward Suh, Yevgeniy Vorobeychik, Zhuoqing Mao, Somesh Jha, Patrick McDaniel, Huan Sun, Bo Li, and Chaowei Xiao. Autodan-turbo: A lifelong agent for strategy self-exploration to jailbreak llms, 2025. URL **https://arxiv.org/abs/2410.05295**.

Mantas Mazeika, Long Phan, Xuwang Yin, Andy Zou, Zifan Wang, Norman Mu, Elham Sakhaee, Nathaniel Li, Steven Basart, Bo Li, David Forsyth, and Dan Hendrycks. Harmbench: A standardized evaluation framework for automated red teaming and robust refusal, 2024. URL https://arxiv.org/abs/2402.04249.

Ninareh Mehrabi, Palash Goyal, Christophe Dupuy, Qian Hu, Shalini Ghosh, Richard Zemel, Kai-Wei Chang, Aram Galstyan, and Rahul Gupta. Flirt: Feedback loop in-context red teaming. *arXiv* preprint arXiv:2308.04265, 2023.

OpenAI. Openai api. **https://openai.com/api/**, 2024. Accessed: 2024-08-28. Anselm Paulus, Arman Zharmagambetov, Chuan Guo, Brandon Amos, and Yuandong Tian. Advprompter: Fast adaptive adversarial prompting for llms. *arXiv preprint arXiv:2404.16873*, 2024.

Maya Pavlova, Erik Brinkman, Krithika Iyer, Vitor Albiero, Joanna Bitton, Hailey Nguyen, Joe Li, Cristian Canton Ferrer, Ivan Evtimov, and Aaron Grattafiori. Automated red teaming with goat:
the generative offensive agent tester, 2024. URL **https://arxiv.org/abs/2410.01606**.

Xiangyu Qi, Yi Zeng, Tinghao Xie, Pin-Yu Chen, Ruoxi Jia, Prateek Mittal, and Peter Henderson.

Fine-tuning aligned language models compromises safety, even when users do not intend to! arXiv preprint arXiv:2310.03693, 2023.

Salman Rahman, Liwei Jiang, James Shiffer, Genglin Liu, Sheriff Issaka, Md Rizwan Parvez, Hamid Palangi, Kai-Wei Chang, Yejin Choi, and Saadia Gabriel. X-teaming: Multi-turn jailbreaks and defenses with adaptive multi-agents, 2025. URL **https://arxiv.org/abs/2504.13203**.

Qibing Ren, Hao Li, Dongrui Liu, Zhanxu Xie, Xiaoya Lu, Yu Qiao, Lei Sha, Junchi Yan, Lizhuang Ma, and Jing Shao. Llms know their vulnerabilities: Uncover safety gaps through natural distribution shifts, 2025. URL **https://arxiv.org/abs/2410.10700**.

Mark Russinovich, Ahmed Salem, and Ronen Eldan. Great, now write an article about that: The crescendo multi-turn llm jailbreak attack, 2025. URL **https://arxiv.org/abs/2404.** 01833.

Xinyue Shen, Zeyuan Chen, Michael Backes, Yun Shen, and Yang Zhang. " do anything now":
Characterizing and evaluating in-the-wild jailbreak prompts on large language models. In Proceedings of the 2024 on ACM SIGSAC Conference on Computer and Communications Security, pp.

1671–1685, 2024.

Chongyang Shi, Sharon Lin, Shuang Song, Jamie Hayes, Ilia Shumailov, Itay Yona, Juliette Pluto, Aneesh Pappu, Christopher A. Choquette-Choo, Milad Nasr, Chawin Sitawarin, Gena Gibson, Andreas Terzis, and John "Four" Flynn. Lessons from defending gemini against indirect prompt injections, 2025. URL **https://arxiv.org/abs/2505.14534**.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion:
Language agents with verbal reinforcement learning. Advances in Neural Information Processing Systems, 36:8634–8652, 2023.

Alexandra Souly, Qingyuan Lu, Dillon Bowen, Tu Trinh, Elvis Hsieh, Sana Pandey, Pieter Abbeel, Justin Svegliato, Scott Emmons, Olivia Watkins, and Sam Toyer. A strongreject for empty jailbreaks, 2024. URL **https://arxiv.org/abs/2402.10260**.

Together AI. Together ai api. **https://www.together.ai**, 2024. Accessed: 2024-08-28.