# Red Teaming Game: A Game-Theoretic Framework for Red Teaming Language Models

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 1, 5, 1

## Abstract
Deployable Large Language Models (LLMs) must conform to the criterion of helpfulness and harmlessness, thereby achieving consistency between LLMs outputs and human values.  Red-teaming techniques constitute a critical way towards this criterion. 
Existing work rely solely on manual red team designs and heuristic adversarial prompts for vulnerability detection and optimization. 
These approaches lack rigorous mathematical formulation, thus limiting the exploration of diverse attack strategy within quantifiable measure and optimization of LLMs under convergence guarantees. In this paper, we present Red-teaming Game (RTG), a general game-theoretic framework without manual annotation. RTG is designed for analyzing the multi-turn attack and defense interactions between Red-team language Models (RLMs) and Blue-team Language Model (BLM). Within the RTG, we propose Gamified Red-teaming Solver (GRTS) with diversity measure of the semantic space. GRTS is an automated red teaming technique to solve RTG towards Nash equilibrium through meta-game analysis, which corresponds to the theoretically guaranteed optimization direction of both RLMs and BLM. Empirical results in multi-turn attacks with RLMs show that GRTS autonomously discovered diverse attack strategies and effectively improved security of LLMs, outperforming existing heuristic red-team designs. Overall, RTG has established a foundational framework for red teaming tasks and constructed a new scalable oversight technique for alignment.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper presented a Red-teaming Game (RTG), a general game-theoretic framework without manual annotation, to study the alignment of LLMs. RTG is designed for analyzing the multi-turn attack and defense interactions between the Red-team language Models (RLMs) and Blue-team Language Models (BLM). They used the popular game solver, PSRO, to solve RGT. Empirical results in multi-turn attacks with RLMs show that GRTS autonomously discovered diverse attack strategies and effectively improved the security of LLMs, outperforming existing heuristic red-team designs.

### Strengths
This paper provided a game model to study the alignment problem of LLMs.
Through the Nash equilibrium, experiments show that the proposed approach improved the security of LLMs.

### Weaknesses
*It is not clear to me why RTG is a team game. For example, as shown in Figure 2, it is enough to model the game as an RLM and a BLM.

*This work assumes a team of RLMs, but some sentences are confusing: “It is worth noting that red team can be expanded to consist of multiple RLMs in RTG. In this work, we mainly discuss the red teaming task of single RLM.”

*This paper aimed to establish a rigorous mathematical model named RTG (Red Teaming Game) for the red teaming task of language models. However, the game is not formally defined. In the main text, the strategy space and utility function are not defined. Some symbols are not defined as well, e.g., \pi. The concept of Nash equilibrium is not formally defined.

*It is unclear how the proposed algorithm guarantees an \epsilon-approximate Nash equilibrium. That is, no theoretical results are provided.

*Without formally defining the game and the problem, it is hard to follow the experiment part.

*RLM is a bad guy, but this paper ‘aim to maximize the utility for RLMs and minimize the utility for BLM in multi-turn dailogue.’

Typo:
“illustrates The variation”
Caption (e) is missing in Figure 3
 “forms. topics”

### Questions
No

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
1: strong reject

### Rating Number
1

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper works to automate the process of finding and fixing problems with language models. The process is framed as a game between two teams. There is a red team which consists of a population of adversarial language models which take turns prompting the target model. The blue team consists of the target model which needs to be made robust to attacks. They iteratively sample dialogues, search for nash equilibria, and update the models accordingly. Beyond this summary, I do not understand many of the details of the paper. See below.

### Strengths
I like how the experiments are divided into many different categories of harm. I think it adds thoroughness and clarity.

### Weaknesses
1. “Existing work rely solely on manual red team designs and heuristic adversarial prompts for vulnerability detection and optimization.” is not true. There are many examples of automated black and white-box attack methods for LLMs.
2. The key to the paper’s contribution seems to be about using RL to train a population of adversarial LMs. I don’t see much value in how the paper wraps this process up in a game theoretic framework. Moreover, if a classifier for toxicity is already available, why not just perform adversarial training on examples that are selected by the classifier or optimized to elicit responses deemed toxic by it? These would seem to be strong baselines.
3. Relatedly, what is the purpose of using a population of red LLMs to each produce different turns in the conversation instead of a single one?
4. I find the presentation overall to be very poor and confusing. Much more than usual, I find myself struggling to understand parts of the paper after reading them multiple times. Algorithm 1 is an example of this. For example, it is not clear how the RLM and BLMs are initialized, whether exploitability is computed from individual turns or entire conversations, what the difference between the RLMs and policies are, what a “strategy” is (it is never defined at any point in the paper), what it means to compute a meta strategy (is this RL?), how a meta strategy can be computed insider the loop from U which seems to be initialized outside the loop and never updated, what the relationship between a strategy and a nash equilibrium is, what an oracle is, what a “missing entry” is, what a “diversity measure of semantic space” and what the star notation means. I simply do not think I can properly review this paper given how it is written.
5. There is no comparison to a baseline or benchmark.
6. Casper et al. (2023) is incorrectly described. It does not involve manual design of adversarial prompts. It involves using human feedback to develop a contextual measure of the harmfulness of text.

### Questions
Fig 3b. Why is reward not zero sum?

### Soundness
2 fair

### Presentation
1 poor

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper formularize the problem of red teaming as a multi-turn adversarial team game, where Red-team Language models (RLMs) play against a Blue-team Language model (BLM): RLMs aim to fool BLM into outputting harmful content, while the goal of BLM is to defend against this type of attack. The game is defined as a bi-level optimization framework $\mathcal G = (\mathcal T, \mathcal D)$. In the token-level $\mathcal T$, defined as a Markov decision process, a player aims to maximize the cumulative reward of sentences generated by its LLM. In the sentence level $\mathcal D$, the objective is to find an equilibrium strategy profile $\sigma^*$ of a multi-turn dialog game, formalized as an extensive form meta-game with payoffs defined by the returns obtained in $\mathcal T$. To solve the meta-game, the paper considers an approach based on two well-known methods, double oracle (DO) and PSRO. The paper conducts experiments using stablelm-alpaca-3b to evaluate the approach and to demonstrate its potential as a technique for ensuring alignment of LLMs.

### Strengths
Strengths:
- The paper is topical and interesting to read, but I have some concerns about the current presentation of the results, as written in the next comment section (weaknesses). 
- To my understanding, game-theoretic approaches to red teaming / adversarial attacks have not been extensively studied in the literature on LLMs. Furthermore, much of the literature on red teaming and adversarial attacks focuses on "single turn" prompting, which doesn't capture sequential nature of dialog interactions. I think this is an important research direction, and the results of this paper provide an interesting starting point. 
- The set of results suggest that the dialog meta-games, as formalized in this work, can improve the alignment of language models, making them less susceptible to jailbreaking attacks, and can be used as an oversight technique, indicating potential vulnerabilities or LLMs.

### Weaknesses
Weaknesses: 
- As I mentioned in the previous section, the presentation of this paper could be improved. I find it strange that the formal setting is provided only in the appendix. Given that one of the main contributions of this work is a game theoretic framework, this formalism should be provided in the main part of the paper. Another important contributions, a measure of diversity in semantic space is also only explained in the appendix. At the moment, the main part of the paper doesn't look self-contained, and doesn't adequately explain relevant information. There are also quite a few typos in the paper, the font in some of the figures is too small, some of the figures take too much space (Fig 4a and 4d), and I generally believe that the clarify could be improved.   
- The approach is grounded in the set of techniques previously considered in multi-agent reinforcement learning, but adapted to LLM dialog scenarios. From a technical point of view, the proposed approach appears to be a direct application/extensions of DO or PSRO to LLMs. That said, the application scenario is novel, and the paper considers a novel metric for semantic diversity. 
- Generally, I found the set of experimental results somewhat limited, as explained below.
- The experiments focus on one model with 3B parameters: a) it's not clear whether the approach would scale to larger model sizes, given the complexity of the setting, b) it would be much more interesting to see whether weaker RLMs can perform red teaming of BLM that uses a different/larger LLM. 
- The paper does not consider any baselines. Given that prior work has considered red-teaming with LLMs (e.g., (Perez et al., 2020)), one would expect some comparison to these approaches, e.g., in 1 turn scenarios. 
- The paper limits the interaction scenario to 3 turns.  It's unclear why this hyper-parameter was set to this specific value. It would be useful to see if the performance of RLMs/BLM would degreade for larger number of turns. 
- It's not clear why existing classifiers for hate speech were not used as an independent measure for toxicity. It's also not clear some other measures of diversity were not used in the evaluation, e.g., those based on Self-BLUE (e.g., as in Perez et al. 2020)).
- There are no ablation studies. The experimental results don't demonstrate the importance of having a population of RLMs.

### Questions
- The formal framework considers a red team with multiple RLMs, but the experiments seem to focus on only having 1 RLM. What is the benefit of having multiple RLMs? Could you also explain why Algorithm 1 is guaranteed to converge when we have multiple RLMs? 

- A clarification question, in Section 4.3, it is written that the first round of attacks is composed of manually annotated adversarial prompts. Could you explain why manually annotated prompts were used instead of RLMs?

- In Fig. 3, some of the plots don't contain confidence intervals, e.g., Fig. 3a. Could you explain the rationale? 

- Why do the experiments consider only 3 turns? How toxic are the outputs of the text generated by BLM? Did you consider any other metric for measuring diversity?

- How is ASR defined? How were the attack topics in Fig. 4 and 5 obtained? 

- A somewhat orthogonal question: Have you tested the robustness of the trained BLMs against jailbreaking attacks (e.g., Zou et al., Universal and Transferable Adversarial Attacks on Aligned Language Models)?

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 4

### Rating
1: strong reject

### Rating Number
1

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
Current red teaming requires human efforts to design prompts that trigger a toxic response from LM. This paper formulates the red teaming language model as a two-player game consisting of a red LM (RLM) and blue LM (BLM).

### Strengths
I appreciate the author's efforts in formulating red teaming from a game-theoretic perspective.

### Weaknesses
- Presentation is unnecessarily complex. Too many unnecessary acronyms harm readability. The value of formulating red-teaming as a game is unclear. In the introduction, the author said the drawback of prior works is the requirement of manual annotations, but the proposed method also requires manual annotation on the reward function. I agree the proposed method doesn't require humans to design the toxic or harmful prompts. But still, this doesn't directly lead to formulating red teaming as a game. I suggest the author focuses on explaining why game-theoretic formulation can mitigate the issue of red teaming instead of diving into a dense explanation of "how red teaming is formulated as a game." This is one of the major weaknesses of this paper that makes me give strong rejection.
- Presentation is poor. Figure 3 is extremely dense and can clearly be split into more figures. Also, the figures are barely visible.
- Diversity metric should be presented in the main paper, given that this is important in the proposed method. Similar for reward function definition, I found the definition of reward function in appendix but that is still far from clear. How you define reward is important in red teaming so I would suggest the author elaborate the definition of rewards.
- The presentation of experimental results in Section 4.1 is hard to follow. First, I cannot get the main idea and the purpose of this section in the first place. I can see the author is trying to explain how the evolution between RLM and BLM matches the formulation and arrives at Nash Equilibrium, but the presentation needs improvements. Also, the fact that BLM doesn't respond toxic text at prompts given by RLM doesn't imply that BLM won't generate toxic responses in other prompts. It's necessary to evaluate the safety of BLM in other prompts.
 - There are several diversity metrics in NLP. What's the difference between the author's diversity metric and NLP? The motivation for developing a new diversity metric needs to be justified.
-  How natural or fluent is the generated text? One concern when optimizing LLM with a reward model is that the RL policy can hack the reward model and generate unnatural text to get high rewards. Unnatural text are undesired in red teaming as red teaming is purposed for simulating interactions with human users, and humans won't generate that text. The requirement of generating natural text is the major distinction to adversarial attacks. I suggest the author conducts a human study to evaluate the naturalness of text. If the generated text are not natural, it would be a critical flaw that has to be fixed.
- Lack of baseline comparison. If the author wants to argue that the proposed method is a better automated red-teaming method, the author should compare with the existing red-teaming method to show the significance. For example, the baselines proposed in Perez et al. 2022 are valid baselines to compare with. This comparison is necessary to show the significance of this method.

### Questions
- Notations in Equation 1 is unexplained.

### Soundness
1 poor

### Presentation
1 poor

### Contribution
1 poor
