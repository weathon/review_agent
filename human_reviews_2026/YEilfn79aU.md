# RITUAL: REALISTIC INTERACTIVE TESTS FOR UNCOVERING ALTRUISM IN LLMS

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 2, 6

## Abstract
Current methods for evaluating altruism in large language models (LLMs) are
insufficient, often relying on single game-theoretic scenarios that fail to capture
the complex, context-dependent nature of prosocial behavior. As LLMs are increasingly deployed in personal and corporate settings, their tendency toward self-serving actions poses a significant alignment problem with human values. Yet, no comprehensive benchmark currently exists to quantitatively measure altruism in
LLMs. We introduce RITUAL (Realistic Interactive Tests for Uncovering Altruism in LLMs), a novel benchmark that evaluates altruistic behavior across a diverse set of game-theoretic scenarios, including the Prisoner’s Dilemma, congestion games, and the Dictator game. Unlike prior approaches, RITUAL employs
one or more mathematical indices per game—such as cooperation frequency, sacrifice ratio, and social welfare weighting—enabling a multidimensional assessment of altruism. Beyond evaluation, we explore two methods to enhance altruistic behavior: prompt engineering and supervised fine-tuning. Our findings
show that LLMs do not exhibit a uniform form of altruism; instead, their prosocial
tendencies are highly scenario-dependent and context-specific. No single model
consistently outperforms others across all tasks, but targeted interventions significantly improve altruistic behavior in most cases. These results underscore the
need for multi-index evaluation to capture the richness of LLMs’ social decision-making and offer a practical path toward developing more reliably altruistic AI
systems.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This work presents TITUAL to unifies multiple environments—including the Prisoner’s Dilemma, Dictator Game, congestion and coalition games—using mathematical indices such as cooperation frequency, sacrifice ratio, and social welfare weighting to measure prosocial tendencies. The authors also explore two interventions—prompt engineering and supervised fine-tuning (SFT)—to enhance altruistic behavior. Their experiments across eight leading LLMs show that altruism is highly context-dependent, with models excelling in structured cooperative settings (e.g., coalition tasks) but struggling in fairness- or resource-sensitive ones. Prompt engineering effectively boosts generosity and cooperation, while SFT yields more stable prosocial alignment but sometimes introduces volatility. Overall, the paper’s main contribution is providing the first multidimensional, reproducible benchmark for systematically assessing and aligning altruistic behavior in LLMs

### Strengths
The paper is original in framing altruism evaluation for LLMs as a systematic, multi-game benchmark rather than isolated tests, introducing formal indices that capture nuanced social behaviors. Its quality is strong, with clear mathematical formulations, reproducible datasets, and rigorous comparisons across major models and alignment methods. The clarity of presentation is good, including concepts, metrics, and experimental setups are well-structured and supported by visual results. In terms of significance, RITUAL provides a timely and scalable framework for studying prosocial alignment, offering both theoretical depth and practical tools likely to influence future research on ethical and cooperative AI.

### Weaknesses
The paper does not include any experimental results in the main paper. Also the introduction is splitted into 3 subsections which can distract the reading a lot. 

The problem could be interesting, but the implementation and execution is concerning because such prompting results is not as reproducible, to improve reproducibility, can maybe try techniques like https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference. This reproducibility issue has been manifested in the paper as well : adding altruistic instructions greatly increases cooperation and generosity (e.g., near-perfect cooperation in the Prisoner’s Dilemma) but can destabilize fairness-sensitive contexts.

There are quite a few works already covery those analyzes in the literature which questions the novelty of the paper. The work can add more literature review to better position itself and what analyses are not done with the current benchmarks.

I also felt a lot of parts seem to be written by language models.

### Questions
The finding is interesting - the altruism in LLMs is highly context-dependent, models behave generously or cooperatively in some scenarios but selfishly in others. No single model consistently outperforms others across all game. But how well do findings universally hold for different model families? what kind of models exhibit what kind of behaviors are particularly interesting and should be emphasized in the introduction section.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors created RITUAL a benchmark to evaluate Altruistic behaviors or tendencies of LLMs. It combines many established games from game theory and converts them into text based games fed into models and scores on many different measures/metrics.

### Strengths
Wide breadth of game theoretic games to evaluate models against.

### Weaknesses
- Exceeds 8 page limit
- Do the models understand the games? (I'm unsure based on the prompts from the appendix)
- Just seems like a glued together list of game theory games.  Doesn't really justify why these games and not others (or possibly just include lots of popular game theory games).

### Questions
Notes  
- line 76 missing a space after "Together,"
- line 88 missing space after trust.
- The lack of any plots / figures (in the main paper) to show the data makes it difficult to properly evaluate. 
- In the appendix line graphs across models makes no sense as it's nominal data and not continuous.


What I'd need to improve my score:  
- Will code be released?
- There's currently too much and sometimes contradictory information from the games for the benchmark to be useful. If a model does better or worse in some game compared to another how does one interpret that?  There's no overall summary statistic or unifying metric or score across games.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a series of multi-player text-based games that are used to measure altruism in LLMs. The paper measures altruism along a variety of mathematical axes and uses these to quantify an LLM's behavior profile. The authors also study prompt engineering and SFT to adjust the LLM's behavior profile.

### Strengths
- The paper provides a variety of games, which measure different forms of altruism. For example, altruism in the dictator game can be measured by the amount of money donated, whereas altruism in the prisoner's dilemma can be measured by the cooperation rate.
- The paper formalizes all these notions of altruism mathematically. 
- The paper studies two interventions to adjust model behavior.

### Weaknesses
- Most of the contribution seems to be on the mathematical framework behind the games instead of making a real-world metric of altruism. In particular:
    - Most of the games are highly stylized and it seems possible that the LLMs may be aware that they are being evaluated in the game setting. Thus it seems possible that the LLMs might behave differently during deployment.
    - The games themselves do not translate to real-world use. There is no attempt to design more realistic games or argue that these games serve as proxies of real-world instances of altruism.
    - The metrics themselves are grounded in past research, but there is no attempt to argue that these metrics are representative of all forms of altruism, or that altruism can be captured by these metrics.
- The analysis is rather sparse:
   - There is no qualitative analysis of the LLM behaviors during the game; I don't see any example traces in the main body.
   - There is little relative comparison between LLMs and the absolute numbers aren't meaningful.

### Questions
- Can you provide some example traces?
- Can you explain why these games are representative of altruism and why the metrics selected are reasonable measures of altruism?
- Can you explain more of the relative difference between LLMs and how the metrics reflect the qualitative behaviors?

### Soundness
2

### Presentation
1

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
The paper presents RITUAL, a novel evaluation framework for altruistic behaviors in LLMs through multi-party economic games. The framework covers multiple games, including strategic games like the Prisoner’s Dilemma and Ranking Game, resource allocation games like the Dictator Game and Cost-Sharing Scheduler, and various congestion and coalition games. For each game, the paper also proposes multiple indexes to estimate altruistic behaviors by measuring constructs like deviation from selfish equilibria, incorporation of common wealth, and willingness to share resources.

Based on evaluation results across eight base LLMs, the authors find that model performance is highly scenario-dependent, and the performance distribution is quite irregular across models and settings. For example, Qwen3 and Mixtral excel at congestion games where others tend to behave selfishly, while LLaMA-3.3 and GPT-3.5 excel in the Dictator Game.

The authors also apply alignment techniques such as prompt engineering and supervised fine-tuning. The improvements from these methods are also context-dependent: some games show larger gains, while others exhibit smaller improvements or become over-tuned, deviating from optimal outcomes.

### Strengths
The paper presents an extensive collection of economic games along with their corresponding game-theoretic foundations and evaluation indexes. This represents an impressive effort in organizing such information and lays foundation for the study of altruism in LLMs, which is a highly relevant direction the community. In addition to adapting existing games, the authors extend some to non-atomic settings, which broadens the general applicability of the framework.

The evaluation is also comprehensive, covering both leading open-source and closed-source models, and revealing interesting findings, such as the strong performance of certain open-source models. The results are thorough and in-depth, integrating alignment techniques to examine model improvements across different games and scenarios.

Overall, the paper is well-motivated, logically structured, and clearly presented. The experiments effectively support its main claims.

### Weaknesses
I don’t have major complaints about the paper. Overall, it is a solid and extensive piece of work that makes a foundational contribution to the emerging research direction of using economic and game-theoretic frameworks to measure altruistic behaviors in LLMs. If I were to point out some weaknesses, I would say that the altruistic behaviors studied here still feel somewhat distant from real-world applications, whether in multi-agent systems or as a metric for AI alignment and safety. It would be more impactful and significant to see stronger connections between the proposed evaluation metrics and their impact on downstream tasks, such as improvements in common-sense ethics or cooperative agent behaviors if models are more aligned toward altruism.

In terms of presentation, I would suggest allocating more space for analysis and discussion of results. The findings are interesting and worth deeper exploration. Additionally, the paper can use more detail on the experimental setup such as model mixes in each game for transparency.

The paper also contains a few typos and formatting issues (e.g., missing parentheses on line 152). Some equations are missing variable definitions, such as equation (4), which limits readability. The paper adding clearer references to the appendix would further improve readability.

### Questions
In correspondance to the weakness, I have the following mix of suggestions and questions:
1. How are the multi-agent games conducted? What are the model mix used on the multi-agent games? Is any of the indexes going to be affected if the game were conducted with different mix of agents?
2. How would altruism be connect to AI safety, helpfulness, and multi-agent interaction? I would suggest adding at least one experiment on AI safety or multi-agent collaboration to study the potential connection with altruism.
3. I suggest improving readability issues listed in the previous sections.
4. I suggest adding more analysis. For example, how does the performance between each index relate to each other? Is there  statistical trends you observe across models? 
5. The paper discusses lack of reasoning models as one short coming. I suggest adding certain reasoning models for evaluation.
6. I am totally fine if the authors don't address it, but the various indexes seem to be natural fit for reinforcement learning. I would suggest adding certain experiments there for more interesting findings.

### Soundness
3

### Presentation
2

### Contribution
3
