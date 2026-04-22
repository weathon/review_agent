# Towards Sustainable Investment Policies Informed by Opponent Shaping

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
Addressing climate change requires global coordination, yet rational economic actors often prioritize immediate gains over collective welfare, resulting in social dilemmas. InvestESG is a recently proposed multi-agent simulation that captures the dynamic interplay between investors and companies under climate risk. We provide a formal characterization of the conditions under which InvestESG exhibits an intertemporal social dilemma, deriving theoretical thresholds at which individual incentives diverge from collective welfare. Building on this, we apply Advantage Alignment, a scalable opponent shaping algorithm shown to be effective in general-sum games, to influence agent learning in InvestESG. We offer theoretical insights into why Advantage Alignment systematically favors socially beneficial equilibria by biasing learning dynamics toward cooperative outcomes. Our results demonstrate that strategically shaping the learning processes of economic agents can result in better outcomes that could inform policy mechanisms to better align market incentives with long-term sustainability goals.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies the InvestESG model (ICLR 2025) and provides both theoretical and empirical evidence that the Advantage Alignment Algorithm (ICLR 2025) - an opponent shaping algorithm - scales better and finds more efficient equilibria than common baselines PPO, and MAPPO. More formally, the paper equips the original InvestESG model with a parameter that it terms 'climate responsiveness parameter' and shows that a simplified version of this augmented InvestESG model exhibits characteristics of a social dilemma (unlike the general case, as claimed in the original InvestESG paper).

### Strengths
Originality: the paper applies an opponent shaping method to InvestESG showing that this algorithm outperforms baselines in terms of scalability and equilibrium selection. Also, if I understand correctly, the paper rectifies the claim that the original InvestESG is a social dilemma for all parameter configurations. Thus, originality comes essentially from the synthesis of two recent ideas.

Quality and Clarity: the paper is generally clearly written and rigorous with proofs being correct to the extent that I could verify. The experimental results contain enough detail to be reproducible.

Significance: the paper contributes to the literature about using ML to tackle social problems, and in this case, climate change.

### Weaknesses
The main weaknesses in my opinion are the following:

- While the paper is clear in its presentation, it does not discuss enough its most important modelling decisions and assumptions, i.e., whether the model of a stochastic game is indeed adequate to capture climate change and whether the climate responsiveness parameter is rich enough to capture something valuable. In particular, a value of \(\alpha=70\) seems to correspond to a rather unrealistic setting: \(\lambda\) is calibrated to reflect the 1.3 trillion commitment and scaling this by 70 seems to result in a setting of only theoretical interest. Whithout aiming to re-review the original InvestESG model, I think that this paper does not help to lift the limitations of the original model, but rather introduces some more need for discussion based on the points that I mention above. 

- The paper claims as its main contribution that Advantage Alignment outperforms (various variants of) PPO in terms of equilibrium efficiency and scalability. In terms of practical importance, I am not sure what exactly we learn from this in the particular climate change setting. Shall we expect that companies will be prescribed to use the AA algorithm and, thus, achieve better outcomes in (toy model) of battling climate change? Why is opponent shaping relevant in this model and why should a company use (or a government enforce/incentivise its use). Since the paper's intented contribution seems to be the climate change mitigation, I found that the practical relevance of modeling and results was lacking.

### Questions
In addition to discussing the weaknesses above, I was slightly confused about the claim that the original InvestESG model is not a social dilemma for its original parameter configuration. It would be great if the authors could clarify this statement. In particular, does this rectify an inaccurate claim in the original paper where the contrary is claimed?

### Soundness
2

### Presentation
3

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
The paper applies Advantage Alignment (an opponent shaping algorithm) on a varied version of InvestESG environment, a realistic simulation of corporate and investor interactions under climate risk, and compared to baseline algorithms such as IPPO and MAPPO.
Its major contributions include:

1. Formally prove that the parameter $\alpha$ (the climate responsiveness to mitigation) is critical for making the game a social dilemma in the stochastic setting.
1. Prove AdAlign is more effective in finding social welfare maximizing strategies compared to baseline algorithms both theoretically and empirically.

### Strengths
1. **Clear analysis and diagnosis of InvestESG benchmark**: The paper formalizes the conditions when InvestESG is a dilemma. Then it validates the predicted 𝛼 threshold empirically: the single-firm/investor sweep exhibiting a sharp change near 𝛼≈30 and full game behavior at 𝛼=70. The proof is rigorous and the empirical implementation is well-executed. This is very useful to the community who would like to utilize this benchmark for policy analysis and policy making.
1. **Comparison between AdAlign and baselines**: In Section 5.1, the paper compares the final welfare achieved by AdAlign with the result achieved by IPPO and MAPPO, and also gives their interpretation about how opponent shaping alters investment behaviors. The analysis also include inequality analysis. In Section C.3 the paper also addresses the reason why AdAlign is the most applicable option among all the opponent shaping algorithms.

### Weaknesses
**Limited novelty**: The paper focuses on one *existing* benchmark (with some modifications) for deep analysis, and applies an *existing* opponent shaping algorithm on this benchmark. However, the results of original AdAlign paper show that this method has proved to be effective in maximizing social welfare compared to PPO baseline on other benchmarks. Applying the same method to a variation of InvestESG might not considered to be a *fundamental* innovation.

**Asymmetry in comparisons**: The paper claims that AdAlign benefits from self-play (one set of parameters for the company and another set of parameters for investor players) while PPO agents don't. However, it muddies conclusions about algorithmic superiority versus training-regime effects. More ablations (e.g., AdAlign without self-play; PPO with tuned self-play variants) would help.

### Questions
**Real-world implication**: The merit of InvestESG is rooted from its depiction of company-investor interactions in the real world. With the theoretical calculation that $\alpha$=70 leads to a full social dilemma, which makes mitigation much more potent. Can you justify this magnitude in terms of real-world ranges?

**Presentations**: The paper spends Section 2.1, 2.2, 2.3 on formulating Markov Games, RL, and Social Dilemma. The application to Markov games is relatively standard. For better presentation, the paper should focus on its novelty point.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper rigorously analyzes the InvestESG environment. They first formally identify the parameter $\alpha$ required to establish a true intertemporal social dilemma. The authors then apply Advantage Alignment, a scalable opponent-shaping algorithm, to this calibrated environment. This method effectively steers agents toward cooperative, high-welfare equilibria, outperforming standard MARL baselines even without external ESG incentives. The work is supported by a theoretical argument explaining why this opponent-shaping approach is inherently biased toward finding such socially beneficial outcomes.

### Strengths
- The paper's primary strength lies in its formal analysis of the InvestESG simulation. Instead of taking the environment at face value, the authors mathematically derive the precise conditions under which it functions as a true social dilemma.

- The successful application of Advantage Alignment to this complex high-dimensional economic simulation. It demonstrates a scalable method for finding cooperative high-welfare solutions where standard MARL baselines fail.

### Weaknesses
- The paper justifies excluding other OS methods like LOLA or BRS on the grounds of scalability. While reasonable, this means AA is only compared against non-shaping methods (IPPO/MAPPO). It is unclear if AA superior because it's an OS method, or because it's a better OS method. Comparing AA to at least one other OS method on a scaled-down version of the $\alpha$-InvestESG environment would be helpful to make a stronger claim.

### Questions
Your theoretical argument in Section 6 that AA's success stems from a "cooperative bias" induced by GAE critic lag is a central and intriguing claim. However, this mechanism is never empirically demonstrated. Could you provide data from your training runs to substantiate this? For example, could you plot the on-policy mean of the advantage estimates ($b^i$ in your derivation) over time? Seeing this term be positive and decay as the critic converges would provide strong empirical evidence that this is the active mechanism.

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
The authors develop a novel MARL-environment, Invest-ESG, that under certain parameterizations assumes the form of a social dillema. Furthermore, the authors employ Advantage Alignment (AA) for opponent shaping and compare to MAPPO and IPPO baselines. Finally, they provide a theoretical explanation of why yields superior equilibria compared to baselines.

### Strengths
- Originality:
The paper introduces a novel MARL environment that highlights a critical link in broader climate-economic space, namely impact investing and greenwashing risks. Furthermore, the authors make use of SOTA learning algorithms.

- Quality:

The authors are theoretically rigorous in their analysis and include an appendix proving that InvestESG is a social dillema for certain values of lambda along with ablation studies.

- Significance:

Significance largely lies in highlighting the scalability of AA even in a high-dimensional MARL context. Authors provide insight into learned policies through gini coefficient analysis, final mitigation investments, market wealth and climate risk.

Clarity:

Paper is clear and well structured.

### Weaknesses
-The real world impact is overstated. the problem with building international climate agreements is that cooperation is difficult to achieve, using an algorithm that is biased towards cooperative policies doesn’t really capture that phenomena. Missing some connection to actual climate-economic literature about the dynamics of impact investing.

-Theoretical results only valid under strong, unrealistic assumptions.

-Interesting components (greenwashing, resilience investments) of environment disabled.

I will reconsider my score based on the answers to the questions.

### Questions
Q1) Could you include results using AA without disabling greenwashing.

Q2) Have you explored utilizing heterogenous lambda values as certain industrial sectors could impact the likelihood of certain climate hazards? e.g. extractive industries and manufacturing.

Q3) Are there existing econometric models of impact investing that are to some degree comparable with the results seen here?

### Soundness
3

### Presentation
2

### Contribution
3
