## Human Reviewer 1

### Summary
This work aims to encourage cooperation in normal form games through opponent shaping. This is done by proposing Advantage Alignment, where the product of the advantages of the player's value-network and the opponent's value-network are used to update the policy. This means that, when advantages are aligned (i.e., they have the same sign), the log probabilities increase or decrease accordingly. The authors show theoretically that LOLA and LOQA, 2 other opponent shaping algorithms, also follow the advantage alignment principles. They then experimentally validate advantage alignment on Coin Game, Negotiation Game, and the high-dimensional Commons Harvest Open environment.

### Strengths
Strengths:
- This paper follows a long line of work on opponent shaping (LOLA, POLA, COLA, LOQA) and builds on LOQA to propose a new opponent shaping algorithm. While it feels a bit incremental, I find the work very well motivated, and theoretically justified. Extensive experiments on different domains and with relevant baselines confirm its practical performance.

### Weaknesses
Concerns:
- Having access to the opponent's value function results in a specific setting where all the player's preferences are public. The negotiation game completely changes if we know the preferences of the opponent, and we could devise a simple strategy that maximizes the average utility of both players. Since the utilities in this experiment are orthogonal to each other, there does not seem to be any real dilemma. For these reasons, I believe the insights gained from the Negotiation Game experiments to be limited.
- Advantage Alignment uses the product of advantages to align the policy. When there are more than 2 players involved (as in Commons Harvest Open), does the alignment depend on the product of the advantages of all players? If so, if a single agent does not cooperate, the whole alignment product fails. This would limit the scaling of Advantage Alignment to $n$ players.

### Questions
Additional clarification questions:
- Fig6 seems to indicate that Advantage Alignment is not really stable (increasingly noisy rewards, partial collapse towards the end of training against AC and AD). Is it sensible to hyperparameters? Or is there another reason for this behavior?
- Fig4 selects the best agent out of 10 seeds. Given the variance observed for the Negotiation Game in Fig6 (which granted is not the same game as Fig4), how representative are the results in Fig4?
- I am curious how Advantage Alignment can play against AD or AC if it can't change its policy (AD and AC are fixed policies). Doesn't this break Assumption 1?


Overall, I recommend for acceptance, for the strengths listed above (theoretically sound, well motivated, experimentally justified).

### Soundness
4

### Presentation
4

### Contribution
2

### Rating
8

### Confidence
4

---

## Human Reviewer 2

### Summary
This paper focuses on opponent shaping.  They propose an opponent shaping formulation that aligns the advantages of interacting agents, thereby simplifying the mathematical formulation of opponent shaping and reducing the associated computational complexity. In this formulation, the probability of future mutually beneficial actions is increased when their interaction has been positive. They demonstrate their methodologies' effectiveness on several social dilemmas and connect opponent shaping's past successes with this new advantage alignment formulation.

### Strengths
This paper is very well-written, and I appreciated how the authors helped the reader build intuition and understanding of the significance of their technique in section 4. The experiments served to reiterate the strength of their algorithm's performance, and the authors gave solid context for why each environment was selected.

### Weaknesses
The main concern I have with the paper is its individuality from the LOQA work, which concerns a similar technique, applied on similar problems, that achieves similar results. Both the more complex experiments (negociation and and harvest open) and the attached proofs in the appendix helped differentiate some of the beneficial aspects of Advantage Alignment in terms of scalability.

### Questions
- For the coin game, in the always defect case, LOQA receives slightly higher return than Advantage Alignment. Do you have any ideas for why this is?
-  In figure 3, there is mention of green values to show cooperation, but I do not see the green values in the figure.
- In Appendix 6, you mention that equation 59 uses the sum of past advantages of the agent up to the current time step and the advantage of the opponent at the current time step. Did you experiment with much tuning for number of terms to include in the sum of past advantages or number of future terms?

### Soundness
3

### Presentation
4

### Contribution
3

### Rating
8

### Confidence
2

---

## Human Reviewer 3

### Summary
This paper investigated a long-standing research problem in multi-agent systems and game theory, named social dilemma. To address this problem, this paper stood on the side of opponent shaping and proposed a method that attempted to align advantages of various agents, to achieve a compromise between social welfare and self-interests. The main assumption lies in the connection between the opponent's policy gradient and the agent's policy parameters. Building on this, this paper proposed a paradigm called Advantage Alignment, and a corresponding RL algorithm based on PPO. This paper also unified other approaches of opponent shaping, such as LOLA and LOQA, in the paradigm of Advantage Alignment. The experimental results on various tasks showed the effectiveness of the proposed Advantage Alignment.

### Strengths
1. This paper proposed an original idea that derives the advantage alignment to implement the policy gradient with respect to the opponent (the opponent shaping term), which can reduce the computational complexity. This paradigm has been shown to have connections to previous opponent shaping approaches, which is a progress in this research direction.
2. The general quality of this paper is good. Although there are some technical points that I require some furtehr clarifications, most of proofs are correct to my best knowledge. The theorem proofs are actually dependent on assumptions raised in this paper, and assumptions are provided with explanations. The experiments were conducted on sufficient banchmarks, in comparison with multiple baselines. Furthermore, the results not only include the numeric results, but also involve some demonstrations of test case, which make the paper more comprehensive. 
3. This paper is well written. It has a clear motivation in Introduction and a thorough introduction of social delimma and opponent shaping. For those people who are not doing research in this direction, it is still friendly enough for them to catch up. 
4. As for the significance of the research problem, social delimma, it is surely a significant problem. The main reason from my own perspective is that it can simulate a class of social problems. About the opponent shaping direction, the significance to me is sceptical, since the requirement of opponent's knowledge seems like a bit strong assumption. However, I do not deny that this is a necessary step towards the weaker and more realistic assumptions. As a result, this paper is significant within the resesrch community of opponent shaping.

### Weaknesses
I have some technical concerns about this paper, which are specifically listed as follows:
1. In line 720, could you explain why the $\beta$ term is missing?
2. In line 739, could you give more details about how equation (16) is derived from equation (15)?
3. In line 755, could you give more details about how to transform from (19) to (8), step by step?
4. In line 773, could you give more details about how equation (24) is derived from equation (23)?
5. In line 783, even with the assumption of orthonormal gradients, it is still not clear why equation (27) can be derived from equation (26). Could you please clarify this step by step?
6. In line 792, is $A^{*}_{\text{LOLA}}$ equal to $V^{1}$?
7. In line 809, could you explain why $\alpha \nabla_{\theta_{2}} V^{2}(\theta_{1}, \theta_{2})$ is equal to $\Delta \theta_{2}$?
8. In line 1001, there is a typo: equation equation -> equation.

Additionally, I also have some concerns about the experimental analysis:

9. In the results shown in Figure 1b, could you give some intuitive interpretation on the asymmetry between the results of CD and DC? For example, is it related to your theoretical claims?

### Questions
See concerns in weaknesses. If the these concerns can be resolved, I will consider to raise the score. However, for the moment I can only give a reject due to those concerns.

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
4

---

## Human Reviewer 4

### Summary
The paper proposes a novel method for opponent shaping based on the advantage functions of the interacting agents. The derivation unifies prior works through the lens of advantage alignment. Empirically, the proposed method is effective in learning complex coordination behaviors under various multi-agent environments.

### Strengths
- The proposed derivation is novel, and provides novel insight on opponent shaping methods
- The paper is theoretically grounded and well motivated.
- The resulting derivation unifies prior works nicely (Theorem 1 and 2).
- Broad suite of baselines and evaluation environments.

### Weaknesses
- I wish Section 2 and 3 are more detailed, and cover more technical background on prior works. The additional information would allow the reader appreciate the contribution much better.
- The evaluation protocol is not provided. For readers that have little background in opponent shaping, it is not clear if the agent is trained prior to the evaluation or being trained (and thus adapted) during the evaluation episodes.
- It is not clear why increasing the log prob when both agents' advantages are negative is desirable (Eq. 8 and Fig. 1a).
- To my understanding, the proposed method does not outperform the baselines LOQA algorithm. The authors should discuss the differences and why it is desirable to use the proposed method over LOQA.
- The paper claims that the proposed method is more efficient than the baseline but does not elaborate on this aspect.

Minor comments:
- The advantage function is not defined mathematically.
- I believe Eq.2 is not the right expression, though I think this does not affect the main derivation of the paper.
- The connection between Eq.5 and Eq.6 is not explained.

### Questions
- Could the authors explain how LOLA and LOQA and AdAlign are related? I see the derivation in Theorem 1 and 2 but do not understand the implication and how they differ in terms of their training objectives.
- It is not clear why increasing the log prob when both agents' advantages are negative is desirable (Eq. 8 and Fig. 1a). Can the authors elaborate on this?

### Soundness
3

### Presentation
2

### Contribution
3

### Rating
8

### Confidence
2