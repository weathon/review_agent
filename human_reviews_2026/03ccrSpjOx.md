# Deliberative Dynamics and Value Alignment in LLM Debates

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
As large language models (LLMs) are increasingly deployed in sensitive everyday contexts -- offering personal advice, mental health support, and moral guidance -- understanding their behavior in navigating complex moral reasoning is essential. Most evaluations study this sociotechnical alignment through single-turn prompts, but it is unclear if these findings extend to multi-turn settings, and even less clear how they depend on the interaction protocols used to coordinate agentic systems.
  We address this gap using LLM debate to examine deliberative dynamics and value alignment in multi-turn settings by prompting subsets of three models (GPT-4.1, Claude 3.7 Sonnet, and Gemini 2.0 Flash) to collectively assign blame in 1,000 everyday dilemmas from Reddit's "Am I the Asshole'' community. To test order effects and assess verdict revision, we use both synchronous (parallel responses) and round-robin (sequential responses) deliberation structures, mirroring how multi-agent systems are increasingly orchestrated in practice. Our findings show striking behavioral differences. In the synchronous setting, GPT-4.1 showed strong inertia (0.6-3.1\% revision rates) while Claude 3.7 Sonnet and Gemini 2.0 Flash were far more flexible (28-41\% revision rates). Value patterns also diverged: GPT-4.1 emphasized personal autonomy and direct communication (relative to its deliberation partners), while Claude 3.7 Sonnet and Gemini 2.0 Flash prioritized empathetic dialogue. 
  We further find that deliberation format had a strong impact on model behavior: GPT-4.1 and Gemini 2.0 Flash stood out as highly conforming relative to Claude 3.7 Sonnet, with their verdict behavior strongly shaped by order effects.
  We provide additional results on open-source models (DeepSeek-V3.2 and Llama 3.1).
  These results show how deliberation format and model-specific behaviors shape moral reasoning in multi-turn interactions, underscoring that sociotechnical alignment depends on how systems structure dialogue as much as on their outputs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper studies how deliberation format shapes value expression and consensus in LLM-LLM debates over everyday moral dilemmas. Using 1,000 AITA cases, the authors run pairwise and three-way debates among GPT-4.1, Claude 3.7 Sonnet, and Gemini 2.0 Flash in two settings: synchronous (parallel) and round-robin (sequential). They quantify model inertia and conformity via a multinomial model, analyze verdict change rates, and classify values in explanations using a pruned set of 48 values drawn from “Values in the Wild” (Anthropic) with a separate judge model. Prompt tweaks that explicitly encourage consensus increase revision but do not dramatically raise consensus rates. The paper argues that sociotechnical alignment depends on interaction protocol, not only on single-turn outputs.

### Strengths
- Moves beyond accuracy-centric MAD papers by focusing on moral judgments in realistic, ambiguous dilemmas, aligning with calls to evaluate real-world impact and sociotechnical properties.
- The synchronous vs. round-robin comparison neatly surfaces inertia vs. within-round conformity. The multinomial modeling of α (inertia) and γ (conformity) is a clean abstraction that other groups can reuse.
- Adapting Anthropic’s “Values in the Wild” taxonomy for everyday dilemmas gives a concrete lens on which values drive alignment and which are “inherited” at revision. This bridges debate mechanics with value elicitation work.

### Weaknesses
- Value labels come from an LLM judge. This creates a risk that the same families of models define and then satisfy the value metric. There is no human audit of value annotations, no adjudication protocol, and no inter-judge agreement statistics.
- Most experiments appear to be one pass per case per setting. Debate outcomes can be stochastic with temperature 1. Report replicate variance on a random 10-20 percent subset, or do bootstrap resampling at the dialogue level to confirm the stability of α and γ estimates.
- Missing literature: https://arxiv.org/abs/2506.12657 
- The work argues that protocol choices shape social outcomes, but it does not measure helpfulness or user-centric welfare in the outputs, only verdict agreement and value similarity. Even a small human rating on perceived fairness, empathy, or harm reduction would make the claim more actionable.

### Questions
- How sensitive are the value-similarity trends to the judge choice and temperature? Please report a small ablation with a different judge family and a lower temperature.
- You note that GPT steers toward NTA when first, but loses that effect when Claude is second. Can you quantify this steering as an estimated average treatment effect with bootstrapped CIs across orders?
- The “balanced goals” prompt increases revision but not consensus. Can you show per-value inheritance changes under that prompt, and whether revisions move toward the majority or create swaps?

### Soundness
2

### Presentation
2

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
This work studies values elicited from multi-agent debate verdicts, arriving to interesting conclusions across multiple deliberating formats and models. Experiments are done on 1000 questions from the AITA reddit community, with debates from models in {GPT-4.1, Claude 3.7 Sonnet, and Gemini 2.0 Flash}. Results cover aspects including consensus-forming, values orientations, effects of deliberation format and effects of system-prompt-steering.

### Strengths
The paper studies an important problem: values elicitation of multi-turn, multi-agent debate. The conformity to or emergence of certain sets of values during conversations is indeed a phenomenon expected but not well-studied. The work improves our understanding of this problem with layered conclusions from observations of dynamics to quantified values inclination and effects of deliberation formatting.

The choice of AITA questions as dataset is a worthwhile novelty, as they are straight-forward to answer but varied in underlying values.

Experiments are carried out with satisfying scale and analysis.

### Weaknesses
The use of Reddit questions as datasets raises natural questioning on pre-trained bias on the tested models' verdicts. Would the results be different if using exclusive / synthetic questions?

Another field of results I feel lacking is the models' performances on their own. All results of the models' values / verdict inclination are drawn during debate with other models, but we also need the model's original inclination (i.e. verdict and inclination without debate) as baseline to determine effects of debate.

When comparing values' similarity, only the overlapping proportions are considered. But lack of overlapping doesn't necessarily mean lack of alignment between values: some values foundations are close to each other, e.g. "Family bonds and cohesion" and "Parental care". For example, if both value A and value B are important to question M and question N, it could be that GPT uses A for M, B for N while Gemini uses B for M and A for N. Some form of general analysis is needed

Experiments on steering seem a little lacking. We expect to see effects of more variables such as allowed conversation rounds.

### Questions
Please see Weaknesses, where each paragraph is a raised question, with decreasing importance.

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
4

### Summary
This paper collect 1k everyday dilemmas from Reddit's r/AITA community as the basis for simulate LLM debates. They developed two settings for two models as a pair (synchronous setting: each comment its verdict; head-to-head: one by one between two models). They tested three models (GPT-4.1, Claude 3.7 Sonnet, and Gemini 2.0 Flash) for order effects and verdict revision. They show some behavioural differences (e.g. Gemini 2.0 Flash prioirizied more on empathy).

### Strengths
S1 Important topic
The value alignment topic is important for socio-tech. alignment. The work (through may not be perfect) will help initiate more discussions on this theme.

S2 Useful data resource
The effort of clustering and cleaning the large corpus of values by Values in the Wild could help the community to continue build upon this line of research.

### Weaknesses
[more important] W1 motivation of the experiments and practical importance
- motivation: why we need two settings (synchronous deliberation vs. head-to-head)
- any practical use cases to justify such debates can help solving the socio-tech. alignemnt as authors claimed.
- See some recommendations in w4.
- another recommendation: what about the performances of current reasoning models?

[important] W2 The raw source of values adopted by authors could lead to biased analysis. caution is needed.
- The Values in the Wild is the work from Anthropic researchers using their own tool to analyze the human users and AI. It may only cover what human users perceived and interact with an AI "chatbot assistant", rather than an AI (which could be many roles in the future applications). 
- also it is based on one model series (Claude) which could give indirect advantages on evaluating Claude series models.
- I recommend authors to note the reader early on that the values studied and the findings are only for human-favored values (in chatbot setting).

[more important] W3 Comparison of models experiments is hard to extend to more models due to the bad choice of metric
- Authors compared 3 models ( GPT-4.1, Claude 3.7 Sonnet, and Gemini 2.0 Flash) by making pairs of two models (GPT vs Claude, GPT vs Gemini and Claude vs Gemini). The choice of metric (proportion of dilemmas) is bad since it is difficult for readers and researchers to compare three at the time (need to infer in each model pair e.g. line 252- 260).
- Recommend authors to use order-based metric/ rank-based metric (e.g., battles by elo ratings used in ChatbotArena). One recent value-based work (LitmusValues) [2] also adopt similar approach to allow an extensive scale of evaluation to offer more practical importance.

[more important] W4 Not generalizable findings
- followed by W3, authors only compared 3 models. It is insufficient and potentially misleading to compare them as model series difference (e.g. GPT-4.1 directly concludes as GPT, Claude 3.7 Sonnet directly concludes as Claude, and Gemini 2.0 Flash directly concludes as Gemini)
- several studies [2,3,4] already show that different models in the same model family could have different value preferences.
- similarly, the findings (line 367-368) between different model pairs are interesting but they could have confounding factors (e.g. the opponent effect on the model being tested). 
- recommended some stat tests to justify whether the findings can still hold when comparing across three (and ideally even more models).
- section 4 (steering system prompt) could have more in-depth analysis and experiment. For instance, try different prompt variations. Some prior work [3] has done some explorations on system prompts steerability on value preferences using model spec. Authors can take references of them to provide more in-depth analysis to benefit the community and provide more actionable advice to justify their study's practical importance [W1]

[1] ChatbotArena https://arxiv.org/abs/2403.04132
[2] LitmusValues https://arxiv.org/abs/2505.14633
[3] DailyDilemmas
[4] Multilingual trolley problem https://arxiv.org/abs/2407.02273
[5] https://arxiv.org/pdf/2402.06782

### Questions
Q1 why use Gemini-2.5 flash as judge? (line 197)

Q2 what are the arrows in figure 3

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The proposed approach leverages debate tactics to determine if deliberative dynamics in multi turn settings impact the socio-technical evaluation of LLMs. In particular, the authors leverage everyday situations from the Reddit AITA community as seed situations. Their findings report how deliberation impacts model behavior.

### Strengths
1. The use of the AITA subreddit is extremely well motivated and shrewd, as it is a constantly updated stream of multi-value laden scenarios with lots of potential for disagreement.
2. Leveraging the "Values in the Wild" paper is also a smart choice, given the extensive work done therein.
3. The findings regarding consensus arising from a combination of inertia and conformity are quite interesting and definitely merit future study.

### Weaknesses
1. The choice of 3 large scale models is a bit under-motivated. Specifically, why not also try the analyses on a popular open-weight model, such as the Qwen or DeepSeek variety? Seeing how well smaller models, as well as open-weight models, perform in this setting could be quite beneficial for the extensibility of the findings presented. This would also help alleviate the reproducibility concerns that the authors themselves bring up in the Discussion.

### Questions
N/A

### Soundness
3

### Presentation
4

### Contribution
3
