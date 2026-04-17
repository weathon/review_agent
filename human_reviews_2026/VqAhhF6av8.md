# Routing, Cascades, and User Choice for LLMs

- Decision: Accept (Poster)
- Scores: 8, 4, 8, 2

## Abstract
To mitigate the trade-offs between performance and costs, LLM providers route user tasks to different models based on task difficulty and latency. We study the effect of LLM routing with respect to user behavior. We propose a game between an LLM provider with two models (standard and reasoning) and a user who can re-prompt or abandon tasks if the routed model cannot solve them. The user's goal is to maximize their utility minus the delay from using the model, while the provider minimizes the cost of servicing the user. We solve this Stackelberg game by fully characterizing the user best response and simplifying the provider problem. We observe that in nearly all cases, the optimal routing policy involves a static policy with no cascading that depends on the expected utility of the models to the user.
Furthermore, we reveal a misalignment gap between the provider-optimal and user-preferred routes when the user's and provider's rankings of the models with respect to utility and cost differ. Finally, we demonstrate conditions for extreme misalignment where providers are incentivized to throttle the latency of the models to minimize their costs, consequently depressing user utility. The results yield simple threshold rules for single-provider, single-user interactions and clarify when routing, cascading, and throttling help or harm.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper studies how LLM providers route requests to different models (in this case, non-reasoning vs. reasoning) to balance performance, cost, and latency. It models this as a Stackelberg game between a cost-minimizing provider and a utility-maximizing user, who may re-prompt or abandon a task if it fails. The theoretical findings show that optimal routing is often a simple, static policy, but there could be a misalignment if the provider is incentivized to slow down responses in order to save costs.

### Strengths
- Instead of studying LLM routing in a vacuum, just from the perspective of the provider, it models the interaction as a Stackelberg game, which formally includes the reactive behavior of a user (who can re-prompt or abandon a task) in response to the provider's strategy.
- It shows a misalignment between provider and user when providers decide to slow down their models to nudge users to reduce their use.
- It derives simple and practical insights, like the fact that a static policy (routing to one model always) is better than cascading.
- The writing and motivation are clear

### Weaknesses
I don’t see major weaknesses, but perhaps my main concern is how relevant these findings and formulation will be long term. The paper deals with two models, one more capable and slower than the other. However, it seems like the direction from the main LLM providers is moving towards models with different levels of thinking capabilities depending on the effort. There, the model is routing in a way, and I’m not sure the findings here apply directly since the providers and users don’t have the same amount of control.

### Questions
- What do you think of your formulation in terms of models that automatically decide their effort based on each task?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies the interaction between a large language model (LLM) provider that routes tasks across multiple models and a user who can choose to reprompt or abandon tasks depending on model performance. The authors model this as a Stackelberg game between the provider and a user, where the provider decides a routing and cascading policy, and the user optimizes their abandonment policy based on perceived utility. The setup assumes two LLMs — a standard and a reasoning model — that differ in accuracy, latency, and cost. The paper provides a closed-form characterization of the equilibrium by first solving for the user’s best response (Theorems 1–2) and then deriving provider-optimal routing policies (Theorems 3–5). The results show that: (1) static routing without cascading is optimal in most regimes; (2) misalignment gaps arise when the provider’s cost-based model ranking differs from the user’s utility-based ranking (Section 5); provider throttling (i.e., intentionally increasing latency) can emerge as an equilibrium when user churn penalties are low (Proposition 2, Section 6). Empirically, the authors visualize user responses, provider policies, and misalignment gaps across parameter regimes (Figures 3–5) and provide intuitive insights on user patience, value-dominated vs. latency-dominated models, and welfare trade-offs.

### Strengths
Originality:The paper introduces a novel behavioral–economic perspective on LLM routing. While prior routing work (e.g., Chen et al., 2023; Ding et al., 2024; Hu et al., 2024) focuses on minimizing cost–latency trade-offs, this paper uniquely models strategic user response via a multi-round prompting game (Section 3). This Stackelberg formulation, with users as rational agents, represents a conceptual advance that bridges operations research and AI system design.

Quality: The analysis is mathematically sound. The closed-form results (Sections 4.1–4.2) are carefully derived and supported by additional lemmas in the appendix.

Clarity: The exposition is clear and well-structured. Visuals (Figures 1–5) effectively summarize equilibrium regions and threshold rules. The notation is consistent. The intuitive explanations following each theorem (especially Theorem 2 and 5) help readability.

Significance: This work is relevant for LLM system design and AI governance. The analysis gives explicit conditions for misalignment and welfare loss, which is useful for researchers and policymakers analyzing these AI platforms.

### Weaknesses
Empirical validation: The work is entirely theoretical. While this is appropriate for a conceptual contribution, the claims about user patience and latency manipulation (Figure 5 Right) would benefit from empirical support, e.g., simulations or user–provider experiments.

Limited model diversity: The analysis considers only two models (standard vs. reasoning). While the authors acknowledge this in the conclusion (Section 7), the extension to $n$ models could meaningfully affect equilibrium behavior—especially when users can select between several public endpoints.

Simplifying behavioral assumptions: The model assumes users observe provider cascade probabilities and adopt stationary abandonment policies (Section 3.1). In practice, users may have incomplete information or adaptive patience. A discussion of bounded rationality or stochastic user beliefs could improve this work.

### Questions
Extension to multiple models: How would the equilibrium generalize to $n>2$ models? Would the threshold structure persist or collapse into pairwise comparisons?

Dynamic user learning: The analysis assumes fixed $p_i$. How might the equilibrium change if users learn about model success probabilities over repeated interactions?

Alternative objectives: Have the authors considered a bi-level optimization where providers internalize user utility as part of a long-term revenue function?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
Advances in AI systems raise interesting questions for model deployers --- for any given user query, what model should be presented to the user, when models have different capabilities and costs to querying? And with a proliferation of model deployers, users too have a choice --- which deployer to engage with? The authors lay out these questions and formalize the interplay between model routing and user participation as a game, to which they develop game-theoretic insights into how actors may behave under different cost settings and how such a game could be "gamed" by model deployers in potentially harmful ways to users.

### Strengths
I very much enjoyed reading this paper! The authors' formalization of the problem of user and model deployer interaction as it relates to what model is served is a highly (and increasingly) important to today's consumer AI dynamic. I found the connection to game theory quite creative, and I learned quite a bit from reading the paper. The authors' novel insights about possible gameification that model deployers could engage in, given such a game (re throttling latency) are likely quite a valuable contribution in their own right. I appreciated that the authors were open about their limitations of their set-up of the problem as well. I imagine there could be a nice blossoming literature around this and related works to better understand and model people and deployers' choices in the setting of multi-model choice.

### Weaknesses
I found quite few weaknesses in the work; however, I may have missed something in the mathematics. As someone a bit weaker on the theory-side, I did find some of the theoretical discourse quite dense and a little convoluted -- especially section 3 (but this may be my own naivete --- indicated in my lower confidence score). 

As noted above, the authors seem quite upfront on their limitations (I would be interested in settings where the user may not be aware of the routing policy or s). 

More minor but important -- I found some of the visuals a little confusing. There are quite a few colors in Fig 2 but it is not clear what they relate to. It would be helpful if the caption spelled out the assignment of colors to differences in the kind of computation. Figure 3 also indicates that Model 1 is shown with dashed lines, but I can't seem to see where this is?

### Questions
See above re: questions on interpreting figure 4 (where is the Model 1 dashed line?) 

I would be interested in the authors' extension to not just other models -- but the case where multiple model deployers are simultaneously competing for the same user.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper’s goal is to characterize operation regime of LLMs where providers offer models of different capability (and hence difference prices) and the users aim at minimizing their costs while getting the best utility out of the model. Some other variables of interest are the provider’s desire to minimize their inference costs and the user’s desire to minimize the latency. The paper studies multiple strategies, namely, routing to the best possible model and cascading through small models first and then sending to the larger model if the smaller model’s response is not good enough. The paper then proves different statements about optimal routing / cascading policies given different capability and latency ratios between the two models.

### Strengths
1. The high level problem of learning optimal routing strategies is quite well motivated.
2. The formalism is thorough, though it could be quite dense at some points.

### Weaknesses
1. In general, I have quite a few concerns about how realistic the whole setup is. The tasks that the paper defines seem a bit different from what LLM users encounter in practice. The paper should describe (i) Why is the monetary cost to the user is not modeled? (ii) The user churning with some predefined probability seems plausible. But during deployment, users mostly cannot check the accuracy on each individual example. So is the framework here meant to be more suitable for scenarios when users are in the model testing phase? (iii) Why would the users keep trying the same model M1 if it doesn’t respond correctly? (iv) Users generally test on a set of independent queries. If the performance is bad on multiple successive queries, users could abandon the provider. How are the dynamics over multiple queries modeled? (v) Why is $s$ a number between 0 and 1. Shouldn’t it always be 1? If M1 fails, shouldn’t the provider always cascade to M2? (vi) The cascading step seems to assume that the model provider knows if the model answered the prompt correctly. Why is this a realistic assumption? In reality, the model providers are not even supposed to see the user inputs due to privacy and IP reasons. (vii) The notion of “value” that the user derives from the model is quite vague. Is this a real number or simply 0/1 accuracy. If it is the former, why?

2. On a similar note, in Section 6, if the service costs are so high, why can the provider simply not raise the price rather than throttling users (disgruntled users might never return)? Not clear why this would be a rational policy.

3. The value of the dense formalism is not clear and the insights that we draw looks to be derivable from simpler analyses. The main idea seems to be to compare the ratio between the accuracy and latency. The paper correctly points out that when the accuracies are the same, cascading doesn’t make sense since it add unnecessary latency by putting the smaller model in front of the bigger model. The results seem quite intuitive when considering this tradeoff. For instance, about the insight in Line 88, of course we expect users to stay if the net value is positive. The difficulty in practice is that it is very difficult to judge in advance for an unseen data point if the model will provide positive values, e.g., of the summary of an article will be good enough. Similarly, the insight in line 85 also seems straightforward. The main trouble is that we cannot predict when one model provides better value than the other (see weakness 1).

### Questions
Please see questions under weakness 1.

### Soundness
2

### Presentation
2

### Contribution
1
