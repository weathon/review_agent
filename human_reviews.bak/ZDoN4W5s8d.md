# Lossgate: Incomplete Information and Misaligned Incentives Hinder Regulation of Societal Risks in Machine Learning

- Decision: Reject
- Scores: 3, 3, 3

## Abstract
Regulators seek to curb the societal risks of machine learning; a common aim is to protect the public from excessive privacy violations or bias in models. In the status quo, regulators and companies independently evaluate societal risk. We find that discrepancies in these evaluations can be either a detriment or an advantage for companies. To abide by regulation, a company needs to conservatively evaluate risk: it should train its model such that risk remains below the acceptable threshold-even if the regulator's evaluation returns higher risk measurements. This decreases model utility (up to 8%, in our experiments). Conversely, when the regulator's measurements are consistently lower than theirs, we find that a company can behave strategically and game regulation to train more accurate models. We call this Lossgate, an allusion to Dieselgate in environmental regulation: Volkswagen produced cars that limited their emissions when being subjected to a regulator's emissions measurement. To model incomplete information and the misaligned incentives that explain Lossgate, we leverage game theory. We obtain SpecGame, a model for regulator-company interactions which allows us to estimate the excessive risk that results from the strategic behavior observed in Lossgate. We show Lossgate costs 70–96% higher compared to collaborative regulation in the sum cost for all players.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper studies how the mismatch in incentives and information between regulators and companies affects the quality of resulting machine-learning models. The authors frame this question as a principal-agent problem where the principal is the regulator, and the agent is the company. In this framework, the incentives are misaligned because the regulator is more interested in mitigating undesired social impact (e.g., fairness or privacy violations), while the company is more interested in accuracy maximization due to profit maximization incentives. At the same time, this framework also introduces information asymmetry between the regulators and the company due to the difference in evaluation datasets.

The authors analyze this problem in the following way. The end of Section 2 analyzes the committee-produced models to understand the first-best Pareto front for this problem (CollabReg). Then, the beginning of Section 3 focuses on the case where the incentives of both parties are still aligned, but the decisions are made only by the company. This analysis shows how hidden information might lead to sub-optimal decisions. Section 3.1 instantiates a specific game-theoretic setting of repeated Stackelberg games (SpecGame). Section 3.2 proposes an algorithm for finding the Nash equilibria in this setting (ParetoPlay). Finally, Section 4 empirically analyzes the equilibria produced by ParetoPlay and compares them with CollabReg equilibria. These experiments show that hidden information indeed leads to sub-optimal models in terms of the resulting accuracy and the price of anarchy. Additionally, the authors notice that SpecGame leader has a first-mover advantage, and the ParetoPlay algorithm does not result in privacy and fairness violations even in asymmetric information settings.

### Strengths
- The research question is relevant, given current legislative efforts such as the EU AI act, and well-motivated by the Diselgate example.
- The principal-agent framework is well-suited for the analysis. Its application for ML regulations is novel.
- The misalignment of incentives (in terms of privacy and fairness) and the asymmetry in information (in terms of different evaluation datasets) are well-contextualized in the machine-learning setting.

### Weaknesses
- The paper does not clearly mention the regulator's and the company's informational restrictions even though it focuses on hidden information. Sometimes, these restrictions are discernible from the context, but not everywhere. For instance, at the beginning of Section 3, the paper claims that "the vector objective $\ell(s)$ is evaluated on separate datasets," which suggests that neither the regulator nor the company knows the datasets of each other. However, Section 3.2 states that "regulators must obtain theirs (checkpoints) through a third party (e.g., public data) or the company," which suggests that the company might decide to share their dataset with the regulator and eliminate the penalty associated with hidden information.
- Similarly, even though the evaluation loss is a random variable, the paper does not clearly state which characteristic is important for participants. The analysis at the end of Section 2 suggests that the committee cares only about the empirical average of losses. However, the analysis at the beginning of Section 3 suggests that the company only cares about lower bounds for their estimates and upper bounds for the regulator's estimates.
- The incentives for the regulators in SpecGame are odd. Due to discontinuity in $\hat{s}_\omega$ variable, these incentives suggest that the regulators want the company to violate their restrictions but by a little bit because it will give them a big discontinuous advantage in $err$.

- In addition to that, the regularization penalty of the firm, which the paper interprets as fines, simply disappears in the game. However, I would expect the firm to transfer this fine to regulators, which will increase their utility.
- The actions in SpecGame are also non-standard because the regulators could directly observe and manipulate the company's action $s$.
- It is unclear which information both participants get at each step (e.g., the results of loss evaluations, the calculated Pareto frontiers, etc.).
- Line 6 in the ParetoPlay algorithm does not make sense because it adds a vector to a scalar.
- Line 3 in the ParetoPlay algorithm does not correspond to the text. As it is written now, all participants update their Pareto fronts only once.
- Calibrate procedure in the ParetoPlay algorithm is confusing. First, it is not clear whose data it is using. Second, I do not understand how exactly calibration happens.
- I have concerns about the correctness of Theorem 1. First, the analysis of the company's deviation, $s_r$, considers the case when $s_r$ does not belong to a Pareto frontier. However, the other case was not considered anywhere. Second, referenced Corollary 1 considers sequential *simultaneous* games, while the paper considers sequential *Stakelberg* games. Thus, at first glance, Corollary 1 is inapplicable to the setting of the paper.
- The algorithm for plotting Figure 4 is unclear. The parameter $\Delta \varepsilon$ is not present in SpecGame, while the algorithm for the calculation of Pareto frontiers is not presented.
- Due to the unclarity in exposition in the previous sections, the results of Section 4 are hard to interpret.

### Questions
- What are the informational restrictions in all settings? What do the regulators and the company observe in the SpecGame at each step?
- Which loss statistic is relevant for the regulators and the company?
- Why is the regulators' cost discontinuous? Why do the regulators not benefit from fines paid by the company?
- Why are the regulators allowed to manipulate the company's decisions but not allowed to manipulate penalties for the company in the SpecGame setting?
- Can the authors provide a mathematically correct version of Algorithm 1 and a textual description of the Calibrate procedure?
- Can the authors provide a clearer and/or more correct proof of Theorem 1 to alleviate my concerns about the correctness of this result?
- How was Figure 4 produced? By the application of Equation (5)?

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
Much of fair and privacy-preserving machine learning framing assumes 
that the developer gets to select themselves the fairness and privacy hyperparameters.
However, in practice, there is a significant tension between such social concerns
and the accuracy of the trained model.
As developers mostly profit from accuracy,
they are incentivized to weaken fairness and privacy.
The paper accounts for this, 
by introducing a regulator in charge of enforcing a tradeoff between social concerns
and model accuracy.
This naturally yields a game between the developer and the regulator,
which is empirically analyzed by the paper.

### Strengths
By introducing the regulator, 
and the tension between the machine learning developer and the regulator,
the paper makes an interesting contribution 
towards making responsible AI research findings actionable for regulators.

### Weaknesses
Overall I found the paper poorly written,
and there are many core components that do not seem clearly laid out (as discussed below).
Moreover, I am highly unconvinced by the proof of the main theorem (Appendix D).
While this is likely due to my lack of understanding of key objects,
such as the strategy space and the costs of the two players.

In particular, at time $t$, the action seems to be a choice of $s^{(t)}$
(by the regulator at even times, and by the company at odd times).
I am not even sure whether *any* $s^{(t)}$ is a feasible action.
Now, the authors wrote costs that depend on $s$ (e.g. in Table 1).
However, it is unclear to me on which $s^{(t)}$ the value $s$ refers to.

In particular, Algorithm 1 seems to suggest that there is a single $s$
that both players can (arbitrarily?) affect at the next round.
But then, I do not understand how the regulator can affect the company,
who may simply cancel out any modification by the regulator to set their value of $s$.
Additionally, I find Theorem 1 implausible.
For one thing, Algorithm 1 involves a single gradient ascent-descent at each iteration, 
which is a far cry from any kind of optimal best-response.

What is more, I do not understand why the regulator's cost is defined as a case disjunction
depending on $\hat s_w$ and $b \in \{ \gamma, \varepsilon \}$.
First, it yields a weird noncontinuity at $\hat s_w = b$,
which seems to imply that $\hat s_w = b$ is better than $\hat s_w = b - 0.00001$.
Second, it equation (2) already seemed to suggest a natural form for it.

The lack of clarity of the paper on some of its core components
prevents me from recommending acceptation.

More anecdotically:
- The 8% and 96% figures are the extreme observed values over 6 datasets. 
Reporting them is misleading. 
I think it would be more representative of the work to report that
"in our experiments, a few percents of utility is lost for most datasets"
and that "across six experiments, Lossgate consistently incurs around 80% increased cost".
- From a pedagogical point of view,
I would have found it useful to sketch a figure in the $(\gamma, \varepsilon)$-space,
with level lines of the company and of the regulator's losses.
In particular, this would help illustrate the ParetoPlay Gradient Descent-Ascent algorithm.

### Questions
Can the authors clarify the action spaces and the cost of SpecGame (especially on which $s^{(t)}$ they depend)? I am especially interested in a formal definition of the game in a classical extensive form.

Could the authors explain the rationale behind the discontinuity in the regulator's cost function at $\hat s_w = b$? How does this align with the continuous formulation in equation (2)?

There have been some recent advances in "Proofs of Training" [1].
In particular, in principle at least, 
these proofs could allow the regulator to effortlessly verify
that the company's training used some claimed hyperparameters.
How would the implementation of such verification tools affect the paper's findings?

[1] Kasra Abbaszadeh, Christodoulos Pappas, Dimitrios Papadopoulos, Jonathan Katz:
Zero-Knowledge Proofs of Training for Deep Neural Networks. IACR Cryptol. ePrint Arch. 2024: 162 (2024)

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper investigates the societal risks of machine learning (ML), such as privacy violations and bias, and explores regulatory responses aimed at mitigating these risks. It highlights the complex dynamics between companies, which prioritize model accuracy (often for profit), and regulators, who focus on minimizing societal harms. The study introduces Lossgate, a scenario where companies exploit discrepancies between their risk assessments and regulators' to maximize model accuracy. Through game theory, the authors model these regulator-company interactions in a framework they call SPECGAME, which quantifies how strategic behavior can lead to substantial societal costs—up to 96% more than if the entities collaborated. The study reveals that imperfect information and misaligned incentives between companies and regulators can either force companies to adopt overly conservative models, sacrificing accuracy, or enable them to sidestep regulation, raising societal risk.

### Strengths
There are a couple of merits in the paper. First, it reframes ML regulation through the principal-agent problem framework, effectively capturing the inherent misaligned incentives between regulators and companies. By modeling the relationship as a principal-agent problem, the authors illustrate how these incentives shape strategic interactions and allow for regulatory adaptation using game theory to structure and assess regulatory effectiveness in reducing societal risks. Second, the paper introduces a novel simulation algorithm specifically designed to assess the outcomes of this regulatory game. This algorithm simulates interactions between regulators and companies, allowing researchers and policymakers to identify equilibrium points within the game, or optimal outcomes under varying strategies. Through this, the paper provides a tool to validate game-theoretic predictions and explore equilibrium properties that could inform regulatory guidelines. Third, the authors emphasize the role of incomplete information in regulatory settings, noting that differences in how companies and regulators evaluate risks can lead to significant utility losses. They show that this uncertainty, especially when it comes to privacy or fairness audits, affects model outcomes and could result in over-cautious behavior from companies or, conversely, regulatory evasion strategies.

### Weaknesses
First, the paper claims to differentiate itself by incorporating the regulator’s perspective, yet it lacks clarity on what unique insights this approach adds. Although the game-theoretical analysis is thorough, it fails to offer any novel conclusions, as the findings align largely with expectations. Second, The a principal-agent framework is a classic model extensively studied in economics. The paper appear to be a minor adaptation of existing theories. The game-theoretic approach employed feels like a direct application rather than an innovative twist. Third, The paper attempts to integrate issues of misaligned incentives and incomplete information within a single framework. However, this combined approach appears somewhat divergent from the paper’s stated contribution of focusing on the regulator’s perspective. Moreover, the treatment of information asymmetry is underdeveloped and lacks clear distinction from data uncertainty, which may lead to conceptual overlap and confusion between these issues. This ambiguity detracts from the clarity of the framework, as readers may struggle to separate the distinct roles and impacts of asymmetric information and data uncertainty within the regulatory context. Last, the paper is not well written. Key points become difficult to track, and the narrative loses coherence, which may deter readers from fully engaging with the work. Furthermore, numerous typographical errors and inconsistencies reinforce the sense that the paper needs further refinement for readability and cohesiveness.

### Questions
(1) The author should polish the writing and clearly establish the paper’s main research question at the outset. The specific problem that solving the SPECGAME framework addresses should be well-articulated early on to guide the reader.

(2) The paper should explicitly outline its contributions. It would be beneficial to specify whether the paper’s primary contribution is the new insights derived from the game-theoretic model or whether it offers a practical, implementable approach. This distinction will clarify the paper's intended impact and relevance.

(3) To enhance clarity, the results should be contextualized. For instance, when discussing the privacy budget, the statement that it is “on average 6 lower” is vague. Providing a comparative benchmark or explaining the significance of this value would make the results more interpretable.

(4) Since the paper examines both fairness and privacy risks, it would strengthen the argument to discuss the interaction between these two risks in this framework. A comparative analysis of results when each risk is considered in isolation versus jointly could provide valuable insights into the model’s added value in addressing multiple societal risks simultaneously.

(5) The part of incomplete information is confusing. Without the asymmetric information, there is still data uncertainty issue. It does not seem that the paper could disentangle these two mechanisms. Alternative, the paper should discuss what the conclusion be without the issue of incomplete information, and then discuss what’s different with the presence of incomplete information.

(6) To highlight the model’s practical relevance, the numerical analysis should include a comparison between the game-theoretic approach and existing methodologies. This would provide potential users, especially applied audiences, with a clear view of the added benefits or unique insights offered by the proposed framework.

(7) In the data analysis, it’ll be helpful if you could explain how the confidence intervals are obtained. It seems they are based on 5 simulations may raise questions about statistical robustness.

(8) The paper assumes a simple linear combination when integrating three types of losses into the framework. This seems too simple to be true. It might be helpful to have some justifications. 

(9) In equation (2), the dimensions of the cost function and trustworthy specification bounds appear inconsistent. 

(10) The writing is rough with many typos. For instance, line 176 equ(1), subscript “acc” should be “comp”; line 343, two equations (3) and (3), etc.

### Soundness
2

### Presentation
1

### Contribution
2
