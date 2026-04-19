# Policy Design in Long-run Welfare Dynamics

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 6

## Abstract
Improving social welfare is a complex challenge requiring policymakers to optimize objectives across multiple time horizons. Evaluating the impact of such policies presents a fundamental challenge, as those that appear suboptimal in the short run may yield significant long-term benefits.  We tackle this challenge by analyzing the long-term dynamics of two prominent policy frameworks: Rawlsian policies, which prioritize those with the greatest need, and utilitarian policies, which maximize immediate welfare gains. Conventional wisdom suggests these policies are at odds, as Rawlsian policies are assumed to come at the cost of reducing the average social welfare, which their utilitarian counterparts directly optimize. We challenge this assumption by analyzing these policies in a sequential decision-making framework where individuals' welfare levels stochastically decay over time, and policymakers can intervene to prevent this decay. Under reasonable assumptions, we prove that interventions following Rawlsian policies can outperform utilitarian policies in the long run, even when the latter dominate in the short run. We characterize the exact conditions under which Rawlsian policies can outperform utilitarian policies. We further illustrate our theoretical findings using simulations, which highlight the risks of evaluating policies based solely on their short-term effects. Our results underscore the necessity of considering long-term horizons in designing and evaluating welfare policies; the true efficacy of even well-established policies may only emerge over time.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
In this paper, the authors study how policies can affect long-term welfare in a population. Specifically, they prove that under the survival condition, the Rawlsian policy has greater long-run utility than the utilitarian policy, and under the ruin condition, the utilitarian policy has greater long-run utility than the Rawlsian policy. They demonstrate their findings through numerical simulations.

### Strengths
* The paper is very well-written and easy to follow.

* The paper studies the problem of how policies can affect the long-run welfare of the population, which is important and interesting.

* The paper presents theoretical results that have clear implications. The proof is technically sound.

However, since I have no background in economics, I find it challenging to evaluate the significance of the paper's contributions or the reasonableness of its welfare model. So I choose to assign a positive rating to the paper with a low confidence score.

### Weaknesses
See the "Questions" part.

### Questions
* In the welfare model it is assumed that everyone's welfare will decay over time without interventions. But why is this the case? In the paper [Roy Radner and Michael Rothschild, 1975], such a model is used to describe how a manager would assign his efforts to several tasks. In this circumstance, it is reasonable to have a "decay with time" assumption since if a task is ignored then things may get worse over time. However, I cannot see why such an assumption can be generalized to the model of welfare. Also, the "rich-get-richer" assumption seems at odds with the bounded $f$, $g$ assumption. I think the authors should add more explanations about the modeling choice to make the paper more accessible to readers outside the field. I also recommend the authors use real-world data or existing literature to rationalize such modeling choices. (In the current simulations only the initial state of the model is based on true data, and all other components are hand-crafted.)

* In the paper the author only considers the policy that focuses on one individual at a time. Can the authors comment on what the main technical difficulty is in considering a more general policy that forms a probability distribution on the population?

* In the paper, the authors do not touch on the problem of optimality of policies. It seems the current welfare model can be conceived as an average-reward continuous-state MDP. Since the functions $f_i$ and $g_i$ are assumed to be known, finding the optimal policy is equivalent to solving a planning problem in such an MDP, which is well-studied in the literature. Can the authors give a concise discussion about that?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper studies sequential policy design with the goal of maximising long-term welfare. In particular the authors compare two policy classes—Rawlsian and utilitarian—and identify conditions under which each is optimal. They model the population as N agents, where each agent’s welfare naturally decays without intervention but improves with intervention. Sections 3 and 4 compare the policy classes in terms of long-term population welfare and individual welfare, respectively. Under regularity and survival conditions, the Rawlsian policy outperforms the utilitarian; however, under the ruin condition, this implication is reversed. Section 5 contains experiments with initial welfares drawn from SIPP data and the social welfare evolution supports the theory.

### Strengths
- The paper is well-structured and theoretically rigorous. Theorems 1 and 2 in Section 3 are particularly insightful, establishing necessary and sufficient conditions under which the Rawlsian and utilitarian policies are optimal.
- The individual welfare evolution and policy optimality for Rawlsian, utilitarian, and random policies in Section 4 is also novel, requiring new sub-martingale techniques in the proofs.
- Section 5 includes experimental results that support the theoretical finding and we see that Rawlsian policies though initially leading to a drop, are optimal for long-term social welfare.

### Weaknesses
- The modelling conditions, such as "rich-get-richer" and "poor-get-poorer," are well-explained, but the survival condition (line 255) is difficult to interpret. Is this condition standard from Radner and Rothschild, or is it specific to your framework?
- Clarifying how common the survival condition is would help, as knowing whether the system is in a survival or ruin state is essential to determining whether a Rawlsian or utilitarian policy is optimal.

### Questions
- Please address the questions above.
- A minor question: your theory seems to extend to interventions that allocate resources to more than one agent at each time step $\sum a_i(t) = M$. Is there a reason not to include this in the main body, with \(M=1\) as a special case?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper compares the long-term social welfare resulting from two popular policy frameworks, a Rawlsian policy that focuses on the minimizing the smallest welfare loss and a utalitarian policy that focuses on maximizing immediate welfare gain. The comparison is conducted under a model on welfare decay and intervention return, assuming a survival condition and the Matthew effect. The conclusions are that 1) when the survival condition is satisfied, the Rawlsian policy achieves better long-term social welfare than the utalitarian policy; and 2) when the ruin condition is satisfied, the utalitarian policy becomes better.

### Strengths
The paper puts the welfare policy evaluation in a sequential decision making framework. All models and assumptions are written clearly. The theoretical results are backed with numerical experiments and mathematical proof.

### Weaknesses
The theoretical results rely on several strong assumptions, and need more clarification.

### Questions
## Major comments:

1. I am curious about whether the two essential assumptions for the theoretical results in the paper are realistic. These assumption include 1) the return and decay functions are monotone, 2) the bounds on the return and decay functions are uniform across all individuals. For assumption 1), the paper mentions that the social planner does not need knowledge of $f_i$ or $g_i$. In this case, how does the social planner know whether $f_i$ and $g_i$ are monotone in practice? For assumption 2), the "rich-get-richer" modeling condition also needs to be satisfied. However in reality, the rich gets richer at a much faster rate than other individuals. Is assuming the same upper bound for all indivuals realistic?

2. What is the motivation and advantage for considering max-f and max-g policies, as opposed to the max-fg policy? The paper mentions that the max-d policy only requires partial information about the interventions, which is less costly to measure compared to measuring both return and decay. However, the max-g policy still requires information about the decays, which then offsets the previously mentioned advantage.

3. As for the survival condition, how should a social planner assess whether this condition is satisfied in practice?


## Minor comments:

1. Equation (1): The definition of $\mathcal{F}_t$ appears much later.

### Soundness
3

### Presentation
4

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
This paper studies the behavior of Rawlsian and utilitarian policies in long-run welfare dynamics. They show that under a survival condition, Rawlsian policies are better than the idealized utilitarian approach almost surely in the long run. Under a ruin condition, a utilitarian policy will achieve better long-term social welfare. Simulation results are provided to validate the theories.

### Strengths
This paper is in general well-written and smooth to follow. The theoretical results are strong and the technical proofs seem to be rigorous and well-organized. The simulation results validates the theories clearly.

### Weaknesses
The authors basically leave the survival and ruin conditions unjustified, and the simulation settings are ad-hoc. I would appreciate it if the authors could provide discussion and common example settings where the conditions hold.

### Questions
- In line 195, the author motivates max-U policy with the setting where $f_i + g_i$ are increasing functions. However, this is not enough, right? Say $f_1 + g_1$ and $f_2 + g_2$ are such that $u_1 < u_2$ but  $f_1(u_1) + g_1(u_1) \geq f_2(u_2) + g_2(u_2)  $ (although $f_1 + g_1$ and $f_2 + g_2$ themselves are increasing functions).
- In lines 218 and 219, why the max-g policy is considered a variant of Rawlsian policy?  I mean, $i=\underset{j}{\arg \max }\{g_j(U_j(t))\} $ is still maximizing, rather than minimizing the welfare.

### Soundness
3

### Presentation
4

### Contribution
3
