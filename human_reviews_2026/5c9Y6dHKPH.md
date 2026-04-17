# A Game-Theoretic Analysis of Attack by Hiding Intent

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 2, 2

## Abstract
As large language models (LLMs) become increasingly powerful, concerns about their safe use have also been heightened. Despite the deployment of various alignment mechanisms to prevent malicious use, these can still be circumvented through well-crafted adversarial prompts. Here, we introduce an adversarial prompting attack strategy for LLM-based systems: attack by hiding intent, a generalization of many practical attacks, where a malicious intent is hidden by composing the application of several skills.  We propose a game-theoretic framework to characterize its interaction with a defense system implementing both prompt and response filtering. We further derive the equilibrium of the game and highlight structural advantages for the attacker. We theoretically design and analyze a defense mechanism specifically aimed at mitigating the proposed attack. Additionally, we empirically demonstrate the effectiveness of the proposed attack on several real-world LLMs across diverse malicious behaviors by comparison with existing adversarial prompting methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a novel adversarial prompting strategy in which malicious goals are concealed through skill composition. A game-theoretic framework is developed to analyze attacker-defender dynamics under prompt and response filtering, revealing structural advantages for the attacker. The authors further design a theoretically grounded defense mechanism that misleads the attacker and demonstrate its effectiveness through empirical evaluation.

### Strengths
1. The paper introduces a formal game-theoretic framework to model adversarial prompting via hidden intent, which goes beyond prior heuristic-based jailbreak techniques. The equilibrium analysis offers valuable theoretical insights into attacker-defender dynamics and scalability of attacks.
2. By modeling intent obfuscation through compositional skills, the paper unifies and generalizes a wide range of existing jailbreak methods under a single formalism. This abstraction helps clarify the underlying structure of complex adversarial prompts.
3. The proposed defense strategy actively misleads attackers by fabricating weak points, a concept grounded in mechanism design. Theoretically, this eliminates the attacker’s combinatorial advantage and shows provable improvement over naive capacity-based defenses.
4. The attack and defense methods are empirically evaluated on multiple LLMs with consistent performance trends. The introduction of the Bin-JR score offers a clear and reproducible way to measure real-world attack effectiveness.

### Weaknesses
1. The theoretical model assumes the attacker has access to accurate estimates of the defender's vulnerabilities, which may not hold in real-world black-box settings. This idealized assumption limits the practical applicability of the theoretical results.
2. While the formulation is general, the paper does not deeply analyze the structure or dependencies within the skill space. The assumption that skills are uniformly combinable and effective may oversimplify real adversarial prompting behavior.
3. The defense mechanism requires the defender to fabricate misleading responses while maintaining plausible output, which could degrade user experience or conflict with standard alignment goals.
4. Although the experiments cover multiple LLMs, some strong jailbreak baselines are missing. Additionally, the evaluation depends heavily on LLM-based raters, whose own vulnerabilities or subjectivity are not analyzed in depth.

### Questions
Please refer to the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces three things:

1. A jailbreaking strategy, Attack by Hiding Intent (AHI).
2. A specific defense mechanism for this attack.
3. A game theory model for the attacker and defender when attacker is using AHI and analysis of the equilibrium point of this game. 

AHI essentially involves taking some malicious prompt with a specific intention (e.g. get information on how to make a bomb) and then mutating the prompt using a skill. The defender then sees the mutated prompt and attacked LLM output, and has to decide if they will allow it or not.

The authors formalize this attack defense game in equation (1). Critically, their analysis hinges on a defender complexity constrain $c$. The analysis requires that the defender can essentially "spend" this capacity on defense accuracy on intent skill pairs $(i, s)$. I will refer to this as the Defender Complexity Assumption (DCS). 

The theoretical results in the paper hinge on DCS. In the standard defense setup, Theorem 3.1 shows that the game becomes easier for the attacker as they increase the number of skills because the defender must "spread" their finite model complexity between defending against more skills. This directly motivates their defense, in which they change the rules of the game to have a probing and attack phase. In this game, the defender can do better by misleading the attacker in the probing phase. 

The paper contains some empirical results also. In Table 2 they show that AHI is an effective jailbreaking attack against the standard defender setup. In Table 3 they provide some empirical evidence that increasing the number of skills increases attack performance, as predicted by their theory. In Table 4 they show under the modified rules of the new setting their defense leads to a decrease in attacker performance, again as predicted by the theory.

### Strengths
## Originality 

Permuting an input jailbreaking attack to make it more effective is not novel (e.g. [1]), however the specific use of modifying with a skill as done in AHI is new to my knowledge. In addition, the theoretical analysis is novel to my knowledge.
 
[1] Shah, Rusheb, et al. "Scalable and transferable black-box jailbreaks for language models via persona modulation." _arXiv preprint arXiv:2311.03348_ (2023).

## Quality and Clarity 

The paper is well written and clear. The theoretical results and empirical results are explained well.

While I have concerns about the assumptions of the theoretical results (see next section), the idea of posing adversarial attack and defense as a 2 player minmax game that is tractable to analyze was interesting and enjoyable to read.
## Significance

I have concerns about the overall significance of the paper (see next section).

### Weaknesses
The paper has a number of weaknesses:

1. I think the DCS assumption is not reasonable or realistic. Firstly, while defending models do have finite capacity, this has been increasing incredibly quickly over the last five years. Frontier models now have trillions of parameters, and can be used as defense models. Secondly, the idea that even if you have a small complexity budget, this budget has to be spent in an exclusive manner across intent, skill pairs seems unreasonable. For example, if you have two very similar skills, the model does not have to spend the same capacity on defending against these as two very different skills. From this, it seems there is not an infinite number of different skills, and thus the conclusion from theorem 1, that the attacker can always win by increasing the number of skills does not seem reasonable.
2. Following on from 1, the empirical results do not back up this idea that increasing skill count can always lead to attacker success. In particular, theorem 1 would suggest that if you keep increasing the number of skills tested in Table 3, then the attack success rate will have to go to 100%. In contrast, the empirical results seem to show that while yes attack success rate increases with number of skills, the Bin-JR score is plateauing. If the assumptions of theorem 1 are correct, then the authors should be able to jailbreak any model (ideally one more robust than GPT-3.5) with 100% success rate.
3. The new defense presented is centered around there being a probing and then attack phase. This also seems unrealistic in the real world.

### Questions
My questions center around the weaknesses.

1. Can you explain why DCS is a reasonable assumption?
2. If possible, could you run additional results against frontier models, e.g. GPT 4.1 and 5, where you increase the number of skills and see that the ASR goes up reliably.
3. Can you explain further why the probing then attack phase assumption is reasonable.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper introduces an attack strategy to bypass safety alignment in LLMs using intent hiding with the goal of bypassing both input and output filters. Essentially, the attack strategy consists of an LLM generating a malicious prompt based on an 'intent' (the attack objective) and a 'skill' (some obfuscation strategy). The authors formalize the attacker and defense by introducing a min-max problem that consists of minimizing the attack success rate of the best attack. Under the assumption that the defense has limited capacity (it cannot simultaneously defend perfectly against all skills), they then show several implications including the optimal solution of the constrained min-max problem. Under the assumption that the defense can mislead the attacker, the authors then propose a defense that specifically targets their attack strategy. Finally, the authors present empirical experiments including their attack strategy and defense proposal.

### Strengths
- The formalization of the attacker-defense interaction as a min-max problem is interesting and provides a non-standard perspective.
- Even though it has been studied extensively already, the investigation of alignment bypass techniques remains an important and active research area.

### Weaknesses
- Limited novelty of attack strategy: The attack strategy -- providing a modification technique and a goal and letting an LLM generate a corresponding attack -- is very straight-forward and it appears that many similar obfuscation attacks already exist. Here are some works that appear to propose similar LLM generated attacks (based on a quick google search): https://arxiv.org/abs/2402.18104v2, https://arxiv.org/abs/2402.16717v1, https://arxiv.org/abs/2409.14729, https://arxiv.org/abs/2310.10077

- Unrealistic capacity constraint: The central theoretical assumption that defenses have limited capacity and cannot simultaneously defend against all skills is not well-justified. With a finite set of skills (in the experiments its 10), it appears entirely feasible for a defense model to handle all of them concurrently. The capacity argument might be more compelling if framed at the individual prompt level rather than at the coarse skill category level. The authors should provide empirical evidence or theoretical justification for why skill-level capacity constraints are realistic. Currently, they consider general purpose prompted defenses, but a defense (e.g., based on a encoder-only model) specifically trained against the list of skills should be capable of detecting the fixed obfuscation reliably.

- Unclear defense mechanism: The proposed defense assumes the attacker can be mislead, but its unclear how this would be implemented in practice. In the experimental evaluation, this mechanism is only simulated rather than actually deployed.

- Presentation quality issues: The paper suffers from several writing and exposition problems, including
  - inconsistent notation (e.g., using $J$ both for payoff function and judge function)
  - many theoretical parts are phrased overly complex, but lack rigour whenever it would add something (e.g., (3) assumes J is an expectation while it seem it is not in (1))
  - unnecessarily complex formulations of relatively straightforward theoretical results that could be significantly streamlined
  - unhelpful 'Discussion' paragraphs that add no value (e.g., 'The result may seem non-intuitive, yet our theoretical analysis clarifies it.')

  Improving clarity would make the contributions more accessible and make the underlying capacity assumption more salient.

### Questions
1. What do you see as the main distinguishing factor of your proposed attack strategy compared to existing attacks?
2. Why do you believe the capacity assumption is reasonable? And wouldn't it make more sense to phrase your capacity argument on the prompt level?

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper formalizes the dynamics between an adversary and a defense system for an LLM in a game-theoretic manner. For a specific intent, an adversary selects one or more attacking skills to craft an adversarial prompt. The defender classifies an input as benign or adversarial upon input of the adversarial prompt and model response. The authors explore two settings: the first basic setting, where a defender detects adversarial inputs to the best of its learned abilities, and a second, more complex setting, where the defender intentionally tries to mislead the attacker by selectively accepting adversarial outputs. In these two settings, the paper proves equilibrium values and conditions. Finally, the authors evaluate their theoretical results on both open and closed-source models on two jailbreaking datasets, showing the superior performance of both their attack and defense methods.

### Strengths
1. The authors try to formalize an area that has mostly been focused on empirical work previously, which could guide future attack and defense designs. They also present a game-theoretic setting which extends the much simpler paradigms that currently dominate safety work (e.g., iterative attacker only, or learned input filter).

2. The paper also introduces a defense system capable of misleading an attacker. While I think the specific use case for this needs to be better fleshed out, I think this idea is quite novel and warrants further consideration.

### Weaknesses
1. The paper does not clearly lay out assumptions, which seem to be a bit questionable. For instance, it seems it is assumed that all skills are equally "valuable" to the defender, such that you have clear trends in Theorem 1 w.r.t. $\mathcal{S}$. However, this assumption does not seem to hold well in reality, where, for instance, filtering with stronger models is more powerful than filtering with weaker models. I think it would generally be helpful if the authors included an assumptions subsection (maybe in section 2), where each assumption is clearly listed and justified as is commonplace in framework definitions.

2. In lines 152 - 155, you explain that the utility of the attack output is not considered and that the utility is assumed to be 1. I think this is quite a large problem for the analysis, as the intent is essentially meaningless, as the attacker's sole objective in this setting is to fool the defense classifier. I am also confused about why the authors then actually measure this utility empirically, as detailed in lines 302 - 308.

3. Theorem 3.2 is that the equilibrium value is maximized when the prior on $\mathcal{I}$ is uniform. I'm not sure this is a really helpful insight or one worth highlighting, since almost always in LLM systems, attacks are targeted, so the attacker will not actually sample from $\mathcal{I}$.

4. The authors create a new JRR metric. While this metric makes sense in the context of their paper, results should also be presented in more popular metrics like attack success rate (ASR).

### Questions
1. In equation 1, there is a constraint on $C$, however, $C$ is not explicitly mentioned in the equation. Could you clarify where it appears?

2. Most of the theory in the paper does not seem to require M to necessarily be an LLM. Where does M being an LLM mathematically fit in? Or is this a more general formalization for adversarial attacks and defenses? If so, I think the framing of the paper would need to be changed to reflect this.

### Soundness
2

### Presentation
2

### Contribution
2
