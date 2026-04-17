# Regret-Guided Search Control for Efficient Learning in AlphaZero

- Decision: Accept (Poster)
- Scores: 2, 6, 2, 8

## Abstract
Reinforcement learning (RL) agents achieve remarkable performance but remain far less learning-efficient than humans. While RL agents require extensive self-play games to extract useful signals, humans often need only a few games, improving rapidly by repeatedly revisiting states where mistakes occurred. This idea, known as *search control*, aims to restart from valuable states rather than always from the initial state. In AlphaZero, prior work Go-Exploit applies this idea by sampling past states from self-play or search trees, but it treats all states equally, regardless of their learning potential. We propose *Regret-Guided Search Control* (RGSC), which extends AlphaZero with a regret network that learns to identify high-regret states, where the agent's evaluation diverges most from the actual outcome. These states are collected from both self-play trajectories and MCTS nodes, stored in a prioritized regret buffer, and reused as new starting positions. Across 9x9 Go, 10x10 Othello, and 11x11 Hex, RGSC outperforms AlphaZero and Go-Exploit by an average of 77 and 89 Elo, respectively. When training on a well-trained 9x9 Go model, RGSC further improves the win rate against KataGo from 69.3% to 78.2%, while both baselines show no improvement. These results demonstrate that RGSC provides an effective mechanism for search control, improving both efficiency and robustness of AlphaZero training. Our code is available at https://rlg.iis.sinica.edu.tw/papers/rgsc.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a new technique for improving the efficiency of training of MCTS agents by choosing which states to start playing from, rather than always starting from the very beginning (AlphaZero) or from uniform sampling of visited states (Go-Exploit).

The authors call this method Regret-Guided Search Control (RGSC). It prioritizes sampling states that have high regret, where regret is the discrepancy between the agent's evaluation and the game's actual outcome. To predict such regrets, they train both a regret value network and a regret ranking network. The predictions are then used to select states from a prioritized regret buffer (PRB).

The authors evaluate on a number of games, including Go, Othello, and Hex. They do an ablation study to see whether the regret ranking network actually helps. They also measure whether high-regret states in the PRB gradually decrease during training.

### Strengths
This paper presents a seemingly promising idea of choosing which states to play from, rather than always starting from the very beginning (AlphaZero) or from uniform sampling of visited states (Go-Exploit). Conceptually, it makes a lot of sense that doing so could vastly increase the efficiency of the learning process.

### Weaknesses
According to the plots, the experimental results do not show a strong advantage of the proposed method in terms of Elo rating.

Figure 4 does not show a strong advantage of RGSC over Go-Exploit in terms of Elo rating.

Figure 5 does not show a strong advantage of RGSC over AlphaZero in terms of Elo rating.

Figure 6 does not show a clear superiority of the regret ranking network over the regret value network, especially in 11 x 11 Hex. In the last case, the former *underperforms* the latter for most of the run.

Also, some choices in the design of the ranking surrogate objective have an unclear justification (more comments on this below). It would greatly help if the authors justified each step of their design, as well as how the hyperparameters would be chosen in practice.

Page-by-page corrections:

Page 1:

learning efficient -> learning-efficient

Page 2:

using starting states -> choosing starting states

later training -> later stages of training

Page 5:

redundant parentheses around R(s_j)^(1/τ) in the denominator for the PRB expression

Page 6:

Figures 3, 4, 5, 6: "The shaded area is the error bar with 95% confidence interval" -> "The shaded area is a 95% confidence interval for the mean", if this is what you meant.

### Questions
Page 1:

"represented as cells" What does this mean?

Page 4:

A ranking-based objective might partially address non-stationarity, but can't the ranking (relative order) also change with further training?

Page 5:

Why does the exponential transformation aid optimization? If the aim is non-negativity, why not use softplus or squareplus, which are more stable and less susceptible to overflow for large values?

How do you choose the sampling temperature τ, the EMA rate α, and the probability λ in a principled way in practice?

Appendix 1 describes the hyperparameter values that were chosen. Were these obtained via trial-and-error? How sensitive is the performance to these choices?

The design of outputting the ranking score and getting the resulting restart distribution via a softmax seems a little ad hoc. Can you explain this?

How many trials were used for each figure (3, 4, 5, 6, 7) and table (1, 2)? Were these independent runs of the whole training process?

Page 6:

"All methods use a 3-block residual network" What activation function and weight initialization (presumably He) was used?

Page 8:

Figure 7 lacks uncertainty bands. Was this run for multiple trials?

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
This paper proposes Regret-Guided Search Control (RGSC) to improve the efficiency of AlphaZero-style reinforcement learning. RGSC uses a regret network to identify high-regret states, positions where the agent’s evaluation most diverges from the outcome, and stores them in a prioritized buffer for replay as new starting positions. Experiments on 9×9 Go, 10×10 Othello, and 11×11 Hex show RGSC outperforms AlphaZero and Go-Exploit by 77 and 89 Elo points on average, and it improves win rates even from strong pre-trained models. Overall, RGSC provides an effective mechanism for search control, enhancing both the efficiency and robustness of AlphaZero training.

### Strengths
- Novel and well-motivated idea: The use of regret-guided search control is an intuitive and principled extension of AlphaZero that better mimics human-style targeted learning.
- Clear technical contribution: Introducing a regret network and a ranking-based objective to identify and prioritize high-regret states is a concrete and original addition to the AlphaZero framework.
- Strong empirical results: RGSC shows consistent performance improvements across multiple domains (Go, Othello, Hex) and against strong baselines (AlphaZero, Go-Exploit).
- Robustness to training stage: The method continues to yield improvements even when applied to well-trained models, suggesting it generalizes beyond early training efficiency.

### Weaknesses
- Limited theoretical justification: The paper mainly relies on empirical validation; it lacks a clear theoretical analysis of why regret-based prioritization should improve sample efficiency.
- Scalability concerns: Results are shown only on small board sizes (9×9 Go, 10×10 Othello, 11×11 Hex). It’s unclear if RGSC scales to larger or more complex domains (e.g., 19×19 Go).
- Computational overhead: Maintaining a regret network and prioritized buffer might add non-trivial computational cost; efficiency trade-offs are not clearly discussed.
- Ablation detail: The paper would benefit from more ablation or sensitivity analysis (e.g., effect of buffer size, ranking loss design, or sampling strategy).

### Questions
- Would RGSC remain effective on large-scale domains like 19×19 Go or Atari, where regret estimation might be noisier?
- How significant is the computational overhead introduced by the regret network compared to standard AlphaZero training?
- Could the regret signal be estimated directly from existing AlphaZero components (e.g., policy/value mismatch) without training a separate network?
- How sensitive is RGSC’s performance to the design of the ranking-based loss or the regret buffer prioritization mechanism?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The paper proposes the RGSC, which introduces a regret network to identify high-regret states where the agent's evaluation significantly deviates from the actual game outcome. And experiments are conducted on 9×9 Go, 10×10 Othello, and 11×11 Hex.

### Strengths
1. The paper defines “regret” as the average cumulative deviation from the current state to terminal states. This formulation captures the long-term impact of mistakes while avoiding the locality limitations of single-step error measures. 

2. The method is evaluated on several board-game domains, with a reasonably comprehensive experimental suite within that problem class.

### Weaknesses
well RGSC generalizes beyond these discrete, perfect-information domains. 

2. Dependence on terminal observability: RGSC’s regret computation requires access to complete state-to-terminal outcomes. In many real-world or continuous-control tasks (e.g., robot, Atari), the notion of a single terminal outcome can be ambiguous or trajectories cannot be fully recovered, making the regret signal difficult or impossible to compute reliably. 

3. Baselines: the baselines appear outdated (2023 as the most recent); stronger, more recent baselines should be included to better position RGSC relative to the current state of the art. 

4. Hyperparameter ablation: the method involves many hyperparameters, but the paper lacks ablation studies to show sensitivity and justify the chosen settings.

### Questions
1. Can RGSC be generalized to other benchmarks such as Atari or robotic control tasks where terminal outcomes are ambiguous or partial observability is present?  

2. The method uses many hyperparameters. Have these been validated via ablation studies?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces Regret-Guided Search Control (RGSC), an extension to AlphaZero that enhances learning efficiency in reinforcement learning for board games by prioritizing restarts from high-regret states—positions where the agent's evaluation significantly diverges from the actual game outcome. RGSC incorporates a regret network (comprising a ranking network and a value network) to identify these states from self-play trajectories and MCTS nodes, storing them in a prioritized regret buffer for reuse as starting points.

Contributions include a novel regret-based ranking objective to handle imbalanced and non-stationary regret distributions, a toy example demonstrating regret-guided benefits, and empirical results showing RGSC outperforms AlphaZero and Go-Exploit by 77 and 89 Elo on average across 9x9 Go, 10x10 Othello, and 11x11 Hex. Additionally, RGSC improves performance on nearly converged models (e.g., boosting win rate against KataGo from 69.3% to 78.2%), highlighting its robustness.

### Strengths
The paper demonstrates strong originality by creatively combining search control concepts (e.g., from Sutton & Barto, 2018, and Go-Exploit) with a novel regret-guided prioritization mechanism tailored to AlphaZero. This includes a ranking-based objective for the regret network, which addresses challenges like imbalance and non-stationarity in regret prediction—issues not fully tackled in prior work like Tavakoli et al. (2020) or Trudeau & Bowling (2023). While building on existing ideas, the application to identify high-impact states in MCTS trees and self-play trajectories removes limitations of uniform sampling, offering a fresh formulation inspired by human learning patterns.

In terms of quality, the work is rigorous, with comprehensive experiments on three diverse board games using fixed training states for fair comparison, substantiated by Elo ratings, error bars, and additional analysis. The methodology includes derivations and pseudocode in appendices, enhancing reproducibility.

Clarity is excellent; the paper is well-structured, with intuitive figures.

Significance is high, as RGSC could extend to broader RL domains (e.g., robotics or planning) where sample efficiency is critical.

### Weaknesses
1. Adding the regret network increases model complexity (e.g., extra heads and training objectives), but the paper doesn't quantify the overhead in terms of GPU hours or inference time during self-play. This could be addressed by reporting relative costs compared to baselines, ensuring the efficiency gains aren't offset by higher per-iteration expenses.

2. While RGSC shows final Elo improvements, the Elo curves in Figure 4 reveal that advantages are not consistently stable across training, with baselines often close and fluctuations (e.g., dips around 50K-60K steps) raising concerns about robustness.

### Questions
1. Given the fluctuations and closeness in Figure 4's Elo curves (e.g., overlaps in confidence intervals), how stable is RGSC's advantage? Could you report results from extended training (e.g., to 70K-100K steps) or more random seeds to assess if the lead holds or diminishes over time? 
2. Same question for Figure 6.
3. Could you report the relative computational cost of RGSC (e.g., average time per self-play game or optimization step) compared to AlphaZero and Go-Exploit? How much overhead does the regret network introduce, and does it offset the sample efficiency gains?
4. The toy example uses a simple regret ($|\hat Q(s) - Q(s,a)|$); how does this compare to the definition in Equation 2?
5. I find the RGSC method to be very strong. Could the authors discuss the potential for applying RGSC in more complex domains, specifically those involving stochastic dynamics or partial observability? What challenges would the regret network face in these settings?

### Soundness
3

### Presentation
3

### Contribution
4
