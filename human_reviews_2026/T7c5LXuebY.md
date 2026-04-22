# Market Games for Generative Models: Equilibria, Welfare, and Strategic Entry

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Generative model ecosystems increasingly operate as competitive multi-platform markets, where platforms strategically select models from a shared pool and users with heterogeneous preferences choose among them. Understanding how platforms interact, when market equilibria exist, how outcomes are shaped by model-provider, platforms, and user behavior, and how social welfare is affected is critical for fostering beneficial market environment. In this paper, we formalize a three-layer *model-platfrom-user* market game and identify conditions for the existence of pure Nash equilibrium. Our analysis shows that market structure, whether platforms converge on similar models or differentiate by selecting distinct ones, depends not only on models’ global average performance but also on their localized attraction to user groups. We further examine welfare outcomes and show that expanding the model pool does not necessarily increase user welfare or market diversity. Finally, we design and evaluate best-response training schemes that allow model-provider to strategically introduce new models into competitive markets.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper studies the game-theoretical formulation and its Nash equilibrium (NE) of the "model-platform-user" market exemplified by LLM applications. Specifically, the paper considers the following three-layered market: (1) LLL company releases models, (2) each service platform chooses which LLM to use in their service, and (3) users choose which LLMs to use for their purpose. The paper finds that NE has to conditions that each service uses a different model or all the services use the same model and highlights the finding that increasing the choice of model does not necessarily lead to a social optimum.

### Strengths
- The motivation for studying the "model-platform-user" market is well-explained, and I agree that this is a new and interesting market game to think about.

- The paper provides a solid game-theoretic analysis of the given problem, and the formulation seems reasonable for the most part.

- The paper also presents an algorithm to steer systems to achieve an equilibrium with diverse models, and shows that it works in the experiments.

### Weaknesses
- One discussion I have is whether the best-response dynamics are practical or not. In my understanding, the user and service greedily choose the best model or service for them. However, this may result in a quick change of behavior and does not seem realistic. Recent work on performative prediction considers a gradual change of participants (e.g., Brown et al., 22), and this looks like a more reasonable formulation.

- Another discussion is whether it is reasonable to consider a single reward. This is because services may focus on different tasks such as coding or translation, and depending on the application, what kind of aspects of LLM, may be different from each other. In such cases, I wonder if homogenization or "a-winner-takes-all" dynamics can really happen.

Brown et al., 22. Performative Prediction in a Stateful World. https://arxiv.org/abs/2011.03885

### Questions
- I didn't really understand why score Z can be the negative of S when f is not in A (Def 3.2). I understand that this is needed to show the utility decomposition in Prop 3.3., however, it is not quite intuitive why Z can be anti-proportional when f is not in A.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces a formal game-theoretic model of a three-layer generative AI market, comprising model providers, platforms, and heterogeneous users. It analyzes how platform competition for market share through model selection shapes equilibrium market structures, user welfare, and diversity. The key findings include necessary and sufficient conditions for the existence of pure Nash equilibria (both differentiated and homogeneous), a demonstration that increasing competition (via more models or platforms) does not necessarily improve welfare or diversity, and the proposal of two training methods for model providers to enter such a market strategically. Theoretical results are supported by experiments on synthetic and CIFAR-10 data.

### Strengths
	Originality: The three-layer model formulation is a timely and non-trivial extension of prior work, better capturing the structure of modern generative AI ecosystems.
	Theoretical Rigor: The analysis of equilibrium conditions, linking market structure to the balance between average performance and deviation advantage, is a solid theoretical contribution.
	Holistic Approach: The paper cohesively analyzes the market from platform, user, and (to a limited extent) model provider perspectives, providing a relatively comprehensive initial view.

### Weaknesses
	Strong and Potentially Unrealistic Assumptions: The model relies on several strong assumptions that limit the practical applicability of its conclusions.

Complete Information: The model assumes that platforms and model providers have perfect knowledge of the user type distribution π_θ and their reward functions r_θ(x). This ignores the significant challenge of preference learning and the strategic implications of information asymmetry in real-world markets.

Deterministic User Choice: The hard, winner-take-all user selection rule (Eq. 1) is a simplification. User behavior in practice is often stochastic, influenced by factors beyond a single quality score (e.g., UI, habit, discovery). A softer choice model (e.g., probabilistic) might lead to different equilibrium dynamics and welfare implications.

	Static Nature of Competition:

Non-Strategic Model Providers: The set of models G is treated as exogenous. Model providers are not modeled as strategic agents who dynamically develop or fine-tune models in response to market outcomes, which is a key feature of real-world competition.

Myopic Best-Response: The analysis of platform and entrant strategies primarily considers myopic best-response to static competitors. It does not fully capture the simultaneous, forward-looking strategic interactions where all agents anticipate and react to each other's moves in a multi-stage game, potentially altering equilibrium outcomes.

### Questions
	How would the central conclusions regarding equilibrium existence and market structure change if the deterministic user choice model were replaced with a stochastic one (e.g., a softmax function based on scores)? Could this alleviate the non-existence of PNE in some cases?

	The model relies on the strong assumption of complete information about user preferences. How do you expect the strategic dynamics and your conclusions to change in a more realistic setting with partial or asymmetric information, where platforms must learn user preferences over time?

	In the current framework, when platforms and entrants formulate their best responses, they treat competitors' strategies as static. If we adopt an equilibrium concept where all platforms "act simultaneously" and anticipate each other's reactions (such as Nash equilibrium, which relies on static assumptions in its analysis), do you believe the model entry strategy proposed in Section 5 would remain effective? How should a "strategic entrant," aware that its entry would trigger a recalibration of platform strategies, design its model?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper presents a three-layer framework consisting of the model, platform, and users to study competition in the generative model ecosystem. In this setting, foundation model providers such as Anthropic and OpenAI license their models to platforms like Microsoft Azure and Amazon Bedrock, which then compete to attract users. The authors derive conditions under which a pure Nash equilibrium exists and describe its structure, examining whether platforms use same models (a homogeneous equilibrium) or different ones (a heterogeneous equilibrium). They then analyze user welfare and show that greater competition, either through more models or more platforms, does not always lead to higher social welfare Finally, the paper considers the perspective of model providers and explores training strategies that can improve their overall utility

### Strengths
The main strengths are:
1) Originality in model formulation: The paper introduces a three-layer model, platform, user framework to study competition in generative AI markets. Prior work, typically focuses on two-layer settings involving only users and platforms/models. Their framework captures the distinct incentives at each layer and the competitive interactions (among platforms and among model providers).

2) Section 3 is a highlight of the paper. It provides a clear and rigorous characterization of equilibrium conditions and market structure. The decomposition of platform utility into attraction and deviation components show when platforms converge on a single model versus when they diversify across different ones.

3) Section 5 is also notably original, introducing a strategic perspective on model training. It shows how model providers can adjust their training objectives to improve adoption by competing platforms, with the direct-gradient optimization approach standing out as an innovative method.

### Weaknesses
1. Platforms Limited to a Single Model:

The modelling assumes that each platform selects one model provider. This does not reflect the papers motivation where platforms like Azure and Bedrock host multiple foundation models simultaneously. As a result, the framework cannot capture strategies such as model bundling, which are important for platform differentiation and for covering diverse user needs. This is not a major weakness, but it would be useful if the authors could discuss how multiple-model selection could be incorporated into the current framework.

2. Inconclusive Welfare Insights (Section 4);

The findings in Section 4 are somewhat unclear. Figure 3 illustrates that adding a new model can reduce user welfare, highlighting the “paradox of competition.” At the same time, the authors provide sufficient conditions under which welfare increases. However, it is not clear how frequently these conditions are met in practice, and in the experiments (Section 6) user welfare appears to increase as more models are added. This makes it difficult to draw a consistent, general conclusion about the effect of competition on welfare.

### Questions
- Please address the question raised in the weaknesses, particularly regarding platform model choice. For example, how could the framework be extended to allow platforms to deploy multiple models, and how would this impact the analysis and equilibrium outcomes?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper formalizes GenAI ecosystems as a three-layer market game consisting of model providers, platforms, and user populations. Platforms strategically select genAI models from providers to serve user groups, while users choose platforms based on preferences. The work discusses the structure of equilibria and analyzes conditions for PNE, market differentiation, and user welfare, and provide algorithm for model provider for effective market entry.

### Strengths
The topic is timely and relevant, theoretical analysis is nice and solid. Presentation is easy to follow. Despite the weaknesses I pointed out below, I really like the angle from which the paper formulates the problem and the style of presentation.

### Weaknesses
As a theoretical-oriented paper with the claimed strength lying on the proposed game-theoretic model, my main concern is that the model structure and theoretical finds are not sufficiently interesting for providing new insights. Here is some of my thoughts:
1. The model seem to me is too stylistic. The hardmax user choice model is overly simplified, it would be more interesting to consider softmax or other alternative stochastic choice model and see if similar observation holds. If hardmax is a necessary simplification to derive theoretical results, I believe some discussion or simulations on alternative user behavior models will strengthen the paper. 

2. I do not see a main theoretical claim (there are lots of proposition, lemmas and corollaries, but no theorems). And those results seems to be straightforward and lacks insights. For example, 
 - non-existence of PNE is standard as in my understanding these types of games can only be shown to have PNEs if it has a concave or potential structure. I would expect a deeper understanding like if the PNE does not exist, what would a typical best-response loop look like and how it reflects some phenomenon in reality. 
- Corollary 3.5 conveys a very intuitive message but if unfortunately it rely on a seemingly too restrictive set of assumption. I'm wondering how rare the situation happens when a model has a clear advantage over all other models for the majority user group, while performs similar to other models for all remaining user groups. 
- Lemma 4.2 does not contain any quantitative result. Isn't it trivial to simply saying the welfare at equilibrium can be less than the globally optimal welfare? It is just the definition of the price of Anarchy.
3. The three-layer model does not seem to be tied to the nature of GenAI. The exact same model can be used to study the content creation market as well, if we view the model providers as content creator and candidate models as potential topics or genres of content creation strategies. That said, the proposed model is claimed to capture the market of genAI models but does not actually capture any nuance of the power of GenAI nor how it adds anything special to the market. I might have missed something, if so, I'd appreciate it if the author reemphasize how the competition model is uniquely relevant to genAI.  
4. Section 5 provides little conceptual contribution. I'm not sure how practical the optimization procedure can be actually adopted by a GenAI model provider to determine the new market entry strategy, as it rely on transparent information of all the factors in the market. And one thing it fails to consider is the cost, which should be a very important factor for new players.
5. Limitations in experiments. CIFAR-10 setting is somewhat toy-level and I believe results on language or multimodal models would better reflect generative model ecosystems. Results are lack of statistical significance as well, no error bar seen in figures. And also investigation on the sensitivity of results to various user distributions would be highly appreciated.

Minor issues:
1. typo in Definition 2.1 (Nash Equilibirum) -> Nash Equilibrium

### Questions
Please refer to my questions raised in the weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
2
