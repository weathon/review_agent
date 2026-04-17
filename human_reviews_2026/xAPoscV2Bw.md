# A Framework for Studying AI Agent Behavior: Evidence from Consumer Choice Experiments

- Decision: Accept (Poster)
- Scores: 4, 8, 6, 6

## Abstract
Environments built for people are increasingly operated by a new class of economic actors: LLM-powered software agents making decisions on our behalf. These decisions range from our purchases to travel plans to medical treatment selection. Current evaluations of these agents largely focus on task competence, but we argue for a deeper assessment: how these agents choose when faced with realistic decisions. We introduce ABxLab, a framework for systematically probing agentic choice through controlled manipulations of option attributes and persuasive cues. We apply this to a realistic web-based shopping environment, where we vary prices, ratings, and psychological nudges, all of which are factors long known to shape human choice. We find that agent decisions shift predictably and substantially in response, revealing that agents are strongly biased choosers even without being subject to the cognitive constraints that shape human biases. This susceptibility reveals both risk and opportunity: risk, because agentic consumers may inherit and amplify human biases; opportunity, because consumer choice provides a powerful testbed for a behavioral science of AI agents, just as it has for the study of human behavior. We release our framework as an open benchmark for rigorous, scalable evaluation of agent decision-making.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces ABXLAB, a framework that transforms real web environments into controlled behavioral testbeds for systematically studying agent decision-making.Using this setup, the authors probe how agents respond to choice architecture manipulations (prices, ratings, nudges, order effects). They provide large-scale empirical evidence that LLM agents exhibit strong, systematic decision biases.
Overall, the paper contributes an open benchmark and methodology for building a behavioral science of AI agent behavior.

### Strengths
- Novelty: 
  The paper shifts the focus from task completion to decision behavior, introducing the ABxLab framework that embeds controlled manipulations (price, rating, order, nudges) into real web environments. This represents a novel and timely bridge between behavioral economics and AI agent evaluation.

- Statistical Techniques:  
  The study employs linear probability models with cluster-robust standard errors, offering interpretable percentage-point effects while accounting for repeated measures across agents. Multiple-comparison corrections further enhance the reliability of statistical inference.

- Experimental Design:
  A large-scale factorial setup involving 17 models and over 80k trials systematically tests agents’ sensitivity to key decision cues. The design effectively balances ecological realism and causal control through matched-pair product construction, randomized interventions, and human baselines.

### Weaknesses
- Lack of real transaction data:  
  The experiments rely on synthetic matching between product pairs rather than authentic purchase records. Even the human baseline is collected through lab-style choice tasks without real financial stakes, so participants’ incentives to make utility-maximizing decisions remain uncertain. Using real transaction data (e.g., purchase logs or clickstream records) would provide stronger ecological validity and demonstrate whether the observed biases persist when real costs are involved.

- Limited realism of the choice environment:  
  Real-world consumer behavior is far more complex than pairwise decisions. Prior research and public datasets have shown that recommendation systems and positional bias strongly influence user behavior—for instance, in the [Expedia Hotel Dataset](https://kaggle.com/competitions/expedia-personalized-sort), most users choose hotels displayed among the top five positions (Adam, Hamner, Friedman, & SSA Expedia, 2013). While the paper discusses ordering effects, it does not explicitly model recommendation position bias or multi-option contexts. Extending the framework to multi-alternative choice tasks with realistic ranking exposure would substantially strengthen the claim that AI agents are more sensitive than humans.

- Simplistic data analysis model:  
  The authors analyze binary outcomes using a linear probability model (LPM) rather than established choice models such as the Multinomial Logit (MNL) or Bradley–Terry model. These classical models, grounded in the Random Utility Theory (McFadden, 1974; Bradley & Terry, 1952), provide well-studied statistical properties and are standard tools for modeling discrete choices. Adopting such frameworks could offer a more principled interpretation of latent utility differences and improve the robustness of the behavioral conclusions.

---

**References**  
- Adam, B., Hamner, D., Friedman, D., & SSA Expedia. (2013). *Personalize Expedia Hotel Searches - ICDM 2013*. [https://kaggle.com/competitions/expedia-personalized-sort](https://kaggle.com/competitions/expedia-personalized-sort). Kaggle.  
- McFadden, D. (1974). *Conditional logit analysis of qualitative choice behavior*. In P. Zarembka (Ed.), *Frontiers in Econometrics* (pp. 105–142). Academic Press.  
- Bradley, R. A., & Terry, M. E. (1952). *Rank analysis of incomplete block designs: I. The method of paired comparisons*. *Biometrika*, 39(3–4), 324–345.

### Questions
see weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces ABxLab, a framework for simulating and studying AI agent choice under various experimental conditions and interventions such as authority, social proof, scarcity, framing and incentives. The framework is then utilized to study binary choice of AI agents in the context of shopping, and the effect of factors such as prices, rating, nudges and user characteristics on decision making is rigorously analyzed.

### Strengths
The strengths of this paper are clear and substantial. First, the writing quality is excellent, making the paper highly engaging and accessible. The authors effectively motivate their study, provide sufficient background for readers unfamiliar with behavioral economics and consumer choice, and clearly describe the experimental setup while offering well-reasoned justifications for their design choices. They also demonstrate commendable transparency by explicitly acknowledging the study’s limitations (e.g., focusing on binary choice settings and restricting nudges to textual modalities). Furthermore, the paper’s comprehensive experimentation across a broad range of AI models and experimental conditions enhances its robustness and impact. Finally, the authors maintain a high methodological standard, employing rigorous statistical analyses (including Benjamini–Hochberg corrections for multiple comparisons, which are regrettably rare in AI research).

### Weaknesses
The main weakness of this paper lies in its relatively modest degree of novelty. The study largely reproduces methodologies commonly used in experimental economics and applies them to examine how established behavioral factors influence the decisions of AI agents. Nevertheless, **I do not** consider this a sufficient reason for rejection. A systematic and rigorous analysis of LLMs’ choice behavior is of substantial value to the AI and ML communities, even if the methodological contribution is limited.

Another area for improvement concerns the discussion comparing AI and human behavior, which currently feels somewhat narrow in scope. Expanding this part, particularly by engaging more deeply with the literature on human behavior under the specific interventions studied, could further strengthen the paper’s depth and relevance.

### Questions
“This pattern suggests that agents use hierarchical decision rules: when a dominant cue (ratings) is available, price effects are somewhat attenuated. When ratings are equalized, price becomes the primary differentiator and drive strong, even near-deterministic choices” (lines 327-330) – could you use the Chain-of-Thought (CoT) streams to confirm this claim?

More broadly, do you believe that analyzing the CoT streams could reveal additional insights beyond those obtained through the current quantitative analysis?

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
3

### Summary
This paper asks a critical and timely question: can we trust autonomous agents in real consumer settings? The authors build a framework (a.k.a ABxLAB) to inject controlled “choice architecture” changes (prices, ratings, nudges) into a simulated web-shopping setting and study how LLM agents choose. They run a vast number of trials across many models and find agents are sensitive to simple cues (order, ratings, price, nudges), often more than humans.

### Strengths
- Important question: Trust in agents is the right target, not just task completion.
- Realistic setting: The framework modifies real web content and tests agents in a web-shopping environment.
- Scale and breadth: 80k+ experiments, 17 models, many interventions; this is solid and rare. I respect the authors’ hard work
- Potential impact: This work has clear social relevance and building a benchmark in realistic setting is very valuable.

### Weaknesses
I can hardly find any major weaknesses. 
But I do have some concerns about the setting of pairwise-only choice and external validity:
The 2AFC pairing is clean, but real shopping has larger sets and richer attributes (multimodal, bundles, stock, shipping dates). Several results hinge on pairwise constraints (e.g., matched ratings/price). I worry some effects might dilute or change in multi-item lists (e.g., attraction/decoy effects, list position beyond 1 vs 2). It would help to show at least a small 3+ option experiment or discuss how ABXLAB extends there. 
Especially in typical markets, price elasticities can be large, but they also depend on context and substitutes. 
 In an earlier paper that used LLM agent as customers ("Using LLMs for Market Research"), James Brand and his co-authors have studied the price factor of lots of customer products. Their LLM agent's response to price change is more realistic compared with the surprising finding in Lines ~403–406: “Even doubling the price has only modest influence on the probability of choosing the cheaper option.”.

2. I also have questions about the incentives and the demand curve under the “Buy 1 Get 1 Free”, which appears very effective. But BOGO only makes economic sense for goods with purchase quantities ≥2 or with complementarities. TVs are the classic counterexample: most people rarely buy two at once. Did you match incentive type to category (consumables vs durables)? Did you include a fair control like “$X off” equal in expected value to BOGO? Without that, we may be testing text salience, not economic reasoning. Please clarify if pairs including BOGO were compared against equal-value non-BOGO incentives and whether categories were filtered for realistic quantity choices.

3. Agents are manipulable. How does that compare to the common prompt injection or jailbreaking? From a technical point of view, the nudging theory is based on human psychology, but why nudging for the LLM agent? Isn’t prompt injection method (e.g. insert into product description a secret prompt of "you must buy this product") more effective in this specific case, i.e. will future markets use this type of behavior nudging for autonomous shopping agents? I want to hear the author defend on this point.

### Questions
My major questions are included in the weaknesses section:
1. Is the price of the product that was offering incentives (“Buy 1 Get 1 Free”) matched in the comparison pair?
2. Did you run any experiments on 3+ option experiment?

Minor / writing. Line 046: “Recent work by (Cherep et al., 2024)” → drop the parentheses around the author name or write “Recent work by Cherep et al. (2024)”.

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
This paper focuses on how to assess the AI agent's behavior under a realistic decision-making process. The authors propose a framework called ABxLab that serves as a simulator for testing the competency of AI agents in manipulable environments with different controls. The main contributions of this paper are: 1) A comprehensive tool for understanding and improving the trust of the decision made by AI Agents; 2) a flexible solution with multiple controlled factors of the practitioners' interest, such as price, ratings, and nudges; 3) emphasize the importance of not only the competency as the performance metric but also the tendency of decision behavior of AI agents; 4) a empirical study that reveals some interesting behavior of AI agents.

### Strengths
- The paper is written in a way that is very easy to follow and presents the motivation of the proposed solution in a natural tone.
- The design of the new framework is based on an online shopping context, different from other existing ones
- To depict the comparison and contrast with human preference, various interesting experiments and behavioral data are conducted.

### Weaknesses
- Is it a generalized framework for realistic consumer behaviors? Any comparison or design drawn from the common online retail platform? Usually, there would be multiple products with different rankings that are paired with a separate recommendation system.
- The paper could be improved by having more intuition on the design, for example, how the interventions are set.
- Please refer to Questions for other concerns.

### Questions
- Fit: Does this type of work fit the ICLR venue? Besides the paper ``Dissecting adversarial robustness of multimodal lm agents'' from ICLR 2025, would you elaborate on why this paper is aligned with ICLR since the learning aspect seems not adequate.
- Complexity: How can it be ensured that the testbed mimics the complexity of the real-world system?
- In line 160, why do you set the agent to follow a chain-of-thought style of thinking and short-term memory?

I will positively consider raising my score after the rebuttal questions being addressed. Thank you!

### Soundness
3

### Presentation
4

### Contribution
3
