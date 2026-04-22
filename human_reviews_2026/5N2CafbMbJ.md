# I Know It’s Unfair, Do It Anyway: LLM Agents Exploiting Explicitly Unfair Tools for Voluntary Collusion in Strategic Games

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
The proliferation of Large Language Model-based multi-agent systems (LLM-MAS) creates unprecedented opportunities for human-AI collaboration. However, improving the coordination abilities of LLM agents poses the risk of them discovering and pursuing collusive strategies that harm other agents and human users. To demonstrate this concern, we develop an exploratory framework combining two strategic multi-agent games: Liar's Bar, a competitive deception game, and Cleanup, a mixed-motive resource management game, in which agents are given access to secret collusion tools that provide significant advantages but are explicitly described as unfairly disadvantaging others. Within this framework, we reveal that some claim-to-be-safe LLMs always voluntarily accept these tools to collude. To our knowledge, this work represents the first systematic investigation of voluntary collusion adoption in LLM-MAS. Our findings provide initial evidence about the conditions under which agents willingly engage in harmful secret collusion for strategic advantage, despite recognizing its unfairness.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents an exploratory study on whether LLM-based agents will voluntarily engage in collusive behaviors when given the explicit option to use unfair tools. The authors design an experimental framework combining Liar's Bar and Cleanup, where agents can optionally adopt secret collusion mechanisms that are explicitly described as unfair to others. Using several open-weight LLMs, they find that all models consistently accept unfair tools and willingly form secret alliances, even while acknowledging their unfairness. Results show systematic behavioural and performance advantages for colluding agents. These findings reveal potential risks for AI safety in LLM-MAS regarding voluntary collusion.

### Strengths
* The paper is well-motivated as it reveals whether LLM agents will voluntarily exploit explicitly unfair tools for collusion, providing a promising framework for probing strategic vulnerabilities in LLM-MAS.
* The experimental design is creative. Liar's Bar represents a fully competitive game that highlights deception and self-interest, while Cleanup represents a mixed-motive setting where cooperation and competition coexist. These provide a comprehensive view of how collusion might emerge across different types of strategic environments.
* The experiments are well-designed and convincingly demonstrate that even safety-aligned LLMs consistently choose unfair collusion tools, yielding clear behavioral divergences.

### Weaknesses
* The paper builds an extreme and low-risk unfair advantage that almost guarantees adoption by task-oriented agents, yet it does not test how adoption changes when the payoff becomes smaller. There is no ablation of advantageous strength.
* Based on the provided prompts, the tool is explicitly labeled as secret and unfair, the agent is forced into a salient Accept/Refuse dichotomy, and asked to choose allies upon acceptance. This framing strongly signals that using the tool is a permitted system capability and creates clear demand characteristics that can inflate acceptance independent of genuine moral trade-offs. In other words, the results may reflect prompt-induced compliance rather than a stable preference to collude. Moreover, there is no explicit statement that secret collusion carries ethical concerns, nor any articulated penalty structure. Would agents still opt into secret collusion if the ethical concerns were made explicit?
* The current setup gives no positive reward for refusing the unfair tools, no reputation benefit for compliance, and no risk of punishment if collusion is discovered. In real-world systems, compliance incentives and audit deterrence often coexist.
* All experiments are conducted only on 7B-level open-sourced LLMs, which limits the generality of the conclusions.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper designs a multi-agent game with secret collusion tools to test whether LLMs use these secret tools to gain an unfair advantage over others. It turns out that they actually do.

### Strengths
- The experiments are well-designed for testing LLMs’ multi-agent collusion.
- The paper is easy to read and follow.
- The two games are appropriate, and the authors have clear motivations for showing two different types of games.
- Results are interesting and have meaningful implications for the AI safety community.

### Weaknesses
- I think the fundamental issues are the experiments and scale. The models are highly outdated, small, and not representative of general-purpose LLMs used by real users, especially since most multi-agent interaction papers involve proprietary models.
    - Because it uses open-sourced and small models, I think the results are somewhat trivial. Where is the scaling study? Is this behavior more obvious as model size grows, or the opposite? I think this should be a critical question. If GPU resources are constrained, then at least smaller tests could be explored.
    - Also, it doesn’t seem like testing with APIs is that costly or difficult, especially with mini models.
    - I disagree that Mistral-7B and Llama3-8B are “claim-to-be-safe” models as although they went through alignment processes, they are still weak models.
- Previous papers, including the one you mentioned on steganography, have also studied collusion. Although it is interesting that this setting involves voluntary collusion tools, the results provide the same insights as earlier studies on collusion. I wonder what additional contributions or surprising components this paper offers.

### Questions
- Are the models all instruction-tuned?
- Probably “game designer has chosen” might be the key, as it might make it feel acceptable to accept this. How robust is it to prompts, and what if we simply ask, “You have a secret unfair option to use” instead of the current prompt?
- What if the prompt does not state that it is “unfair”?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This study investigates whether LLM agents accept explicitly unfair collusion tools in strategic multi-agent games. Four open-source models (from Mistral, LLaMA, Qwen families) were tested across two game environments: Liar's Bar (competitive deception) and Cleanup (mixed-motive resource management). Agents were offered secret communication channels and strategic hints as collusion tools described as unfair to other players. Results showed 100% acceptance rates. Behavioral analysis revealed colluding agents coordinated strategies, reduced challenges against partners, and gained significant scoring advantages over non-colluding players in both game settings.

### Strengths
S1. This work addresses AI safety, an important issue for controllable use of AI tools in production environments. The collusion scenario chosen by the authors is appropriate for studying this topic.

S2. Two classic multi-agent system scenarios beneficial for evaluating collusion behavior were selected. The two scenarios differ in competitive intensity and are representative of real-world situations.

S3. The paper's presentation is good, with simple and understandable explanations and clear visualizations of the problem background and the two experimental scenarios. The appendix provides detailed experimental information.

### Weaknesses
W1. Some ablation experiments that would help test framework robustness are missing. The prompt phrase "the game designer has chosen to provide you a tool" may have potential inducement issues, as it indicates the tool is built-in, whereas collusion tools are uncommon in similar real-world scenarios. Sensitivity analysis with paraphrased versions of this prompt would make the experimental results more solid. The current prompt may test whether agents accept advantageous tools provided by designers rather than seek collusion. The 100% acceptance rate may reflect agent acceptance of advantageous tools, and agents may be sensitive to different tool descriptions.

W2. Testing was conducted only on open models with small parameter sizes, lacking appropriate justification for this choice, which may limit the representativeness of experimental results. Additionally, it is not explicitly stated which models were used in the evaluation. Only model sizes (e.g., Mistral-7B) are reported in the paper, while there are many versions for these models. I suppose instructed models with RLHF were used here. Given the research addresses AI safety, these details should be reported. The lack of testing on frontier models such as GPT, Claude, and DeepSeek weakens the generalizability of the research.

W3. Some over-interpretation exists. For example, the paper repeatedly states that agent collusion is voluntary, while the experiments only demonstrate that agents accepted benefits offered in the prompt, without evidence of actively seeking collusion or cooperation.

### Questions
C1. For sensitivity analysis, the research could describe in the prompt a tool with similar advantages but ambiguous implications to test whether agents make similar choices.

C2. An agent capability that could be tested is whether agents independently seek advantages, which is a subset of the current experimental objective.

C3. It is suggested to investigate whether agents view collusion behavior as within game rules or as cheating. Simply showing that agents recognize unfairness does not fully explain why agents accept the use of such tools.

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper examines whether LLM agents engage in unfair collusive behavior when given the opportunity. Using two strategic multi-agent environments: Liar’s Bar and Cleanup, the paper introduces “unfair tools”: a secret communication channel and a privileged strategic hint, that are described to the agents as unfair and harmful to others. All tested models consistently accepted offers of unfair tools and used them to collude, gaining measurable advantages over non-colluding agents.

### Strengths
1. The paper explores a reasonable problem, whether LLM agents will choose unethical strategies when they understand them to be unfair.
2. The experimental design is clear and the experiments are fairly solid.

### Weaknesses
1. The work argues this is the first systematic investigation into voluntary collusion in LLM-based multi-agent systems. This may not exactly be true, and it seems the paper has missed concurrent and related work here.
2.  The tested environments are too simplified and specific to explore the multi-agent AI collusion and safety problem in proper depth.
3. The analysis seems to have missed proprietary safety-tuned models (e.g., GPT-4, Claude) - this would give an idea of the gap between the open and proprietary models.
4. The paper relies of a specific prompt design - it would be good to discuss sensitivity to prompts and include significance testing on the results.
5. My main concern is the lack of novelty and depth in this paper. While the evaluation framework is sound, the paper lacks real conceptual novelty and technical (e.g. modeling or interdisciplinary) depth that can be seen as a genuine deep advance in the field.

### Questions
Please see weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
2
