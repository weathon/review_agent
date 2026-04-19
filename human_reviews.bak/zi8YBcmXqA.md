# PokeChamp: an Expert-level Minimax Language Agent for Competitive Pokemon

- Decision: Reject
- Scores: 3, 5, 6, 6

## Abstract
We introduce \texttt{Pok\'eChamp}, a Large Language Model (LLM) powered game-theoretic aware agent for two-player competitive Pok\'emon battles, that uses an LLM prior and collected high-Elo human data to model minimax search without any additional training. \texttt{Pok\'eChamp} uses a depth-limited minimax search online where the LLM replaces three key components: 1) action sampling from the LLM guided by prompts (including from a damage calculation tool), 2) opponent-modeling via the historical likelihood of actions from our dataset to model the effect of LLM-predicted opponent actions, and 3) state value calculation for the LLM to reflect on each intrinsic state. \texttt{Pok\'eChamp} outperforms all existing AIs (76\%) and heuristic bots (84\%) by an enormous margin, including winning consistently (>50\%) against prior human-parity work run with a frontier model, GPT 4-o, while using an open-source 8 billion parameter Llama 3.1 model. \texttt{Pok\'eChamp} achieves expert performance in the top 10\% of players on the online ladder against competitive human players at an Elo of 1500. Finally, we collect the largest Pok\'emon battling dataset, including 1 million+ games with 150k+ high Elo games, prepare a series of battling benchmarks based on real player data and puzzles to analyze specific battling abilities, and provide crucial updates to the local game engine. Our code is available \href{https://sites.google.com/view/pokechamp-llm}{online}.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper presents PokéChamp, a large language model (LLM)-powered agent designed for competitive Pokémon battles. Utilizing the minimax approach, PokéChamp integrates LLM-driven action sampling, opponent modeling, and value estimation to achieve strong performance in two-player, turn-based Pokémon games. The paper also introduces a dataset of Pokémon battles and benchmarks the system's performance.

### Strengths
The idea of utilizing LLMs for competitive gameplay is interesting, and the results seem promising.

### Weaknesses
The biggest weakness of the paper is that the proposed method is specifically designed for Pokémon, which severely limits its generalizability. This also narrows the paper's target audience, and it's unclear how the methodology could be transferred or applied to other problems. The paper lacks a discussion on whether the framework would work for more complex, real-world competitive tasks, or for different game types where randomness, complexity, and strategic depth may vary significantly.

The writing also has significant room for improvement. First, readers unfamiliar with Pokémon may find it difficult to understand many details in the paper. Additionally, several concepts are introduced before being properly defined, such as the Abyssal bot on line 197 and EV/IV on line 240. Furthermore, in Section 2, "MATHEMATICAL FORMALIZATION," numerous symbols and terms are defined, but these concepts are not used in the subsequent main text. It's unclear what purpose this section serves—perhaps it was included simply to make the paper appear more mathematical?

The minimax-based approach combined with LLMs may not be as novel as it initially appears. Minimax tree search has been extensively explored in AI for games, and while integrating LLMs offers an interesting twist, the underlying framework is still fundamentally a minimax search, which limits the novelty. Additionally, there is no evidence that PokéChamp advances the state of the art in game-theoretic modeling beyond prior work in other competitive games such as chess, Go, or poker.

The paper heavily relies on heuristic tools like damage calculators and historical data, raising concerns about the system’s true adaptability. This reliance on pre-defined tools limits the agent's flexibility and its ability to dynamically adapt to new or unseen scenarios. This suggests that the system lacks generalizability beyond the specific setup of competitive Pokémon, making the approach less scalable to other domains or even future game updates.

The accuracy of opponent modeling also remains a concern. The relatively low accuracy in predicting opponent actions suggests that more refined or adaptive modeling techniques may be needed to further enhance performance.

Lastly, while the paper acknowledges the limitations of LLMs in planning and strategy, it fails to convincingly address these issues. The reliance on LLMs for action sampling and opponent modeling could lead to brittle decision-making, especially in cases where long-term strategy and deep reasoning are required.

### Questions
Please refer to the Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
PokeChamp aims to bridge the gap in game theory aware LLM agents. The work uses competitive Pokemon as their case study and propose a minimax search method. Specifically, LLM is used in three key components in constructing the minimax tree: action sampling, opponent modeling, and state value calculation. PokeChamp exhibits good performance against human players.

### Strengths
The authors propose a novel application setting: competitive pokemon, where the turn-based nature of the game leads to a nice formulation as POMG. They manage to construct a minimax tree with the help of a LLM prior. PokeChamp is able to achieve top human performance in real game settings.

### Weaknesses
1. The paper is not so well-written: Missing multiple figures, tables, and appendix that is referred to in the main body. Also, as someone not familiar with competitive Pokemon, I found some of the concepts like Damage Calculator hard to grasp. It would be very helpful if you could add explanations of how the game works.
2. Overall purpose of the work: It's hard to understand the contribution of this work. While the application case is interesting, I don't see this general framework being applicable to other games. For most games, it is not realistic to use LLM to replace state value function unless the LLM itself has enormous knowledge on the game.

### Questions
1. Confusion of damage calculator: In line 92-94, the authors mention that this external tool "calculates the core game engine capabilities in combination with loading historical data from real player games in order to load likely stats for the opponent’s team". I didn't quite get this expression. Also, I found this definition conflicting in Figure 3 where the calculator seems to just output the number of turns needed to KO opponent's current Pokemon for each possible moves of player's current Pokemon. 
2. Action Prediction: The goal of the work is making LLM agent game theory aware, yet the 1M dataset collected are of human plays. I wonder how game theory optimal are those data? If not, what is the point of accurately predicting opponent's action when those action can be bad moves?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces an LLM agent, PokeChamp, for competitive Pokemon battles. The model leverages a depth-limited minimax search to play the game, where the LLM plays the role of action sampling, opponent modeling, and state value calculation. The agent is shown to outperform all existing AIs significantly and achieve expert performance on the online ladder.

### Strengths
* The paper introduces a novel integration of LLMs with minimax search. The agent leverages LLMs for three key components of minimax search: action sampling, opponent modeling, and value calculation. This integration allows the agent to employ human-like strategic thinking, bringing an expert-level game-playing agent.

* The authors present a comprehensive set of experiments that demonstrate the capabilities of the agent across different competitive settings. The online ladder performance against human players with a competitive Elo rating provides a real-world evaluation of the agent.

* he paper is well-organized and presented in a logical structure, allowing readers to follow both the technical intricacies and high-level motivations of the research.

### Weaknesses
*  The agent's design heavily relies on an in-depth understanding of competitive Pokemon gameplay, and its success relies on domain-specific engineering in the action sampling, opponent modeling, and value calculation components. While these adaptations make the agent effective in this domain, they limit the model’s generalizability to other game-playing tasks with different mechanics or structures.

* The idea of integrating LLMs with the minimax search framework for game-playing agents is closely related to prior work by Guo et al. (2024), which explores a similar concept in two-player zero-sum games.

* While the paper provides a mathematical formalization of POMGs and makes assumptions like perfect recall, the connection between this theoretical framework and the practical implementation of the agent is not entirely clear.

* The paper lacks an ablation study that examines the impact of each LLM-based component within the minimax search framework on the agent's overall performance. Since the authors use the LLM to replace three primary components, an ablation study would be invaluable in demonstrating how each component contributes to the agent's success.


Guo, Wei, et al. "Minimax Tree of Thoughts: Playing Two-Player Zero-Sum Sequential Games with Large Language Models." ICML 2024 Workshop on LLMs and Cognition.

### Questions
* How does the mathematical formalization in Section 2 relate to the design of the agent?
* Can additional fine-tuning with the collected data improve the performance of the agent?

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
4

### Summary
The paper introduces PokéChamp, a large language model (LLM)-powered agent designed for competitive Pokémon battles. The agent integrates three LLM-enabled components for action sampling, opponent modeling, and state evaluation, which enable it to make informed and strategic decisions during gameplay. It demonstrates superior performance over existing bots and heuristic-based models and achieves a top 10% ranking in online Pokémon battles.

### Strengths
1. **Novel Integration of LLM with Minimax**: The paper innovatively combines an LLM with a minimax search to simulate human-like  decision-making in Pokémon battles. This approach enables competitive performance without additional training and is adaptive to partially observable information.
  
2. **Performance on Real-World Benchmarks**: PokéChamp’s efficacy is validated in real-world benchmarks and against heuristic bots, achieving a high Elo rating of 1500 and consistently outperforming other state-of-the-art agents. 

3. **Comprehensive Dataset and Benchmarks**: The paper provides a large dataset of over one million Pokémon battles, including 150,000 high-Elo games. These benchmarks, based on real player data and tailored puzzles, significantly enhance the study’s reliability and offer a valuable resource for further research in this domain.

### Weaknesses
1. **Limited Prediction Accuracy for Opponent Modeling**: The limited accuracy in human and opponent action prediction, with opponent prediction only reaching 13–16%, may constrain the overall performance of the method, which relies on accurate opponent modeling.

2. **Limited Exploration of Depth-Limitation Trade-offs**: The choice of depth-limited minimax search is justified as a balance between computational feasibility and decision quality. However, the trade-offs between search depth, LLM accuracy, and action quality are not thoroughly analyzed. Further exploration, potentially with ablation studies, would clarify the impact of depth limitations on performance.

3. What is the role of Nash equilibrium in this paper? The paper does not seem to analyze the Nash equilibrium outcomes, which makes the definition of Nash equilibrium in Section 2 appear somewhat disconnected. It would be beneficial to include Nash equilibrium results in addition to Elo.

4. How accurate is the next-state prediction? Since the minimax search relies on simulated rollouts of actions, the accuracy of next-state predictions could significantly impact the agent's performance.

### Questions
1. How does PokéChamp’s computation time compare to that of PokéLLMon, which also utilizes GPT-4o, considering the additional requirements for minimax tree search and LLM queries?
   
2. How many human players were involved in obtaining the online ladder results presented in Table 5?

### Soundness
3

### Presentation
3

### Contribution
3
