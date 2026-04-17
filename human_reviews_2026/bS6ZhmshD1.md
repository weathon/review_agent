# Learning from Synthetic Labs: Language Models as Auction Participants

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 0, 2

## Abstract
This paper investigates the behavior of simulated AI agents (large language mod-
els, or LLMs) in auctions, introducing a novel synthetic data-generating process
to help facilitate the study and design of auctions. We find that LLMs reproduce
well-known findings from experimental literature in auctions across a variety of
classic auction formats. In particular, we find that LLM bidders produce results
consistent with risk-averse human bidders; that they perform closer to theoret-
ical predictions in obviously strategy-proof auctions; and, that in a real-world
eBay-style setting, LLMs strategically produce end-of-auction “sniping” behav-
ior. On prompting, we find that LLMs are robust to naive changes in prompts
(e.g., language, currency) but can improve dramatically towards theoretical pre-
dictions with the right mental model (i.e., the language of Nash deviations). We
run 1,000+ auctions for less than $400 with GPT-4o models (three orders of mag-
nitude cheaper than modern auction experiments) and develop a framework flexi-
ble enough to run auction experiments with any LLM model and a wide range of
auction design specifications, facilitating further experimental study by decreasing
costs and serving as a proof-of-concept for the use of LLM proxies.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper examines the behavior of LLMs in simulated auctions and the extent to which their behavior aligns with both theoretical models of optimal bidding strategies and reported human behavior in such environments. The paper considers several experimental settings, including first- and second-price sealed-bid auctions, clock auctions, and an eBay-like auction. The paper examines several intervention forms in the auction framing and studies their effect on deviation from truthful bidding in the (truthful) second-price auction.
 The results indicate that many of the behavioral patterns observed in human participants carry over to LLM participants (e.g., risk-aversion, improved rationality in obviously strategyproof auctions, and end-of-auction 'sniping').

### Strengths
1. The paper offers a rigorous and timely analysis at the intersection of AI, economic theory, and human behavior. It provides valuable insights into how closely LLMs align with theoretical predictions and experimental findings with human participants, highlighting their potential as tools for behavioral economics simulations. In fact, regardless of the degree of alignment with human behavior, I believe that understanding LLM behavior in complex decision-making is crucial, as such models are increasingly deployed as active participants in real-world systems.

2. The paper provides a rich and comprehensive experimental design with multiple auction formats, intervention types, and robustness checks (language, currency, number of players), which strengthens its contribution to the ML community.

### Weaknesses
1. The paper’s literature review is somewhat underdeveloped, making it difficult to situate its contribution within the broader research landscape. Strengthening this section would help clarify its novelty and relevance. In particular, it would benefit from engaging with adjacent lines of work, including:
- studies using LLMs to simulate strategic behavior beyond auction settings [1,2,3];
- research on predicting and interpreting human decision-making with LLMs [4,5];
- alternative approaches to evaluating the rationality of LLMs [6,7].

2. The paper uncovers several interesting behavioral patterns in LLMs, but it does not explore how robust these findings are across prompts, models, and experimental conditions. For example:
- Since the results suggest that LLMs exhibit risk-averse behavior, it would be valuable to test whether these patterns can be influenced through prompting -- either directly (by instructing risk-seeking or risk-averse behavior) or indirectly (by assigning different personas and comparing their risk attitudes).
- Similarly, the observation that LLMs act more rationally in obviously strategyproof auctions raises the question of whether this behavior persists with smaller models or those lacking explicit reasoning capabilities.

(Just to make sure I haven’t overlooked anything — I reviewed Sections E–F for answers to these kinds of questions, but as far as I can tell, those sections focus on other types of robustness checks. Please correct me if I’m mistaken or if I missed any discussion relevant to the points above)

[1] Akata, E., Schulz, L., Coda-Forno, J., Oh, S. J., Bethge, M., & Schulz, E. (2025). Playing repeated games with large language models. Nature Human Behaviour, 1-11.

[2] Shapira, E., Madmon, O., Reinman, I., Amouyal, S. J., Reichart, R., & Tennenholtz, M. (2024). Glee: A unified framework and benchmark for language-based economic environments. arXiv preprint arXiv:2410.05254.

[3] Abdelnabi, S., Gomaa, A., Sivaprasad, S., Schönherr, L., & Fritz, M. (2024). Cooperation, competition, and maliciousness: Llm-stakeholders interactive negotiation. Advances in Neural Information Processing Systems, 37, 83548-83599.

[4] Shapira, E., Madmon, O., Reichart, R., & Tennenholtz, M. (2024). Can llms replace economic choice prediction labs? the case of language-based persuasion games. arXiv preprint arXiv:2401.17435.

[5] Liu, R., Geng, J., Peterson, J. C., Sucholutsky, I., & Griffiths, T. L. (2024). Large language models assume people are more rational than we really are. arXiv preprint arXiv:2406.17055.

[6] Macmillan-Scott, O., & Musolesi, M. (2024). (Ir) rationality and cognitive biases in large language models. Royal Society Open Science, 11(6), 240255.

[7] Alsagheer, D., Karanjai, R., Shi, W., Diallo, N., Lu, Y., Beydoun, S., & Zhang, Q. (2024). Evaluating irrationality in large language models and open research questions. In Proceedings of the HEAL Workshop at CHI. ACM.

### Questions
Do you have an idea whether your findings are robust to variations in temperature and/or the underlying value distribution (i.e., beyond uniform distribution)? In particular, I am curious about the case of temperature = 0, as this choice arguably better reflects the model’s inherent behavioral tendencies.

Additionally, I am open to hearing your thoughts or responses to the weaknesses highlighted above.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
The paper proposes using LLM agents to generate synthetic data for classic auction environments (e.g., FPSB, SPSB, clock/OSP, and an eBay-style setting). The authors claim (1) a flexible, low-cost framework for running many auction instances with LLM “bidders,” (2) qualitative replication of well-known empirical patterns from experimental economics (e.g., behavior under strategy-proof formats and eBay-style sniping), and (3) prompt-level “interventions” that purportedly move agent behavior closer to theoretical predictions. They position the approach as a proof-of-concept toward using LLMs as proxies for human subjects in mechanism design studies.

### Strengths
The question of whether LLM agents can substitute for human subjects in economic mechanism experiments is timely and interesting.

### Weaknesses
1. Poor clarity and factual accuracy: Important auction acronyms are not clearly defined in the Introduction; figures use very small fonts; and at least one factual claim appears misleading (e.g., attributing inspiration to prior work purportedly using GPT-4o where, to the best of my knowledge, Horton23 used GPT-3).
2. Several headline comparisons rely on different granularity and thresholds than the classic studies (e.g., different numbers of bidders, coarser bid grids, broader “match-to-value” criteria). Conclusions such as “similar performance” are therefore not like-for-like and should be qualified or re-analyzed with matched protocols.
3. The “interventions” often reveal dominant strategies or otherwise prime desired responses, making it difficult to attribute improvements to emergent strategic reasoning rather than compliance with instructions.
4. Most results hinge on a single foundation model and a narrow hyperparameter regime (e.g., temperature). There is little evidence that the findings persist across models, decoding choices, or ablations.

### Questions
Can you re-run the principal analyses with exactly matched protocols to the cited human experiments from Kagle93 (same bidder counts, bid grid, payoff reporting, and “match-to-value” thresholds), and report side-by-side metrics?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper demonstrates Language models' ability to mimic human behavior in auction scenarios. They simulate 1,000+ auction experiments and even replicate theoretical results in a quasi-real-world setting.

### Strengths
This paper demonstrates at scale how LLMs can mimic human behavior (or approach theoretical results) in thousands of auction scenarios. I appreciate authors looking into LLM simulations, as it has immense potential in simulating real-world experiments, feeding into the direction of AI-driven science. Overall, the paper is well written.

### Weaknesses
- I completely fail to understand the technical novelty of this paper. Considering the ICLR audience, I do think the papers need to explicitly demonstrate why the proposed framework works as presented. 
- While the finding is exciting (still a bit limited in terms of applicability), it is hard to assess what alternative approach would have worked to achieve similar performance. The work is plagued by a dire lack of reasonable baselines. 
- In fact, as far as I remember, LLM's ability to perform well in an auction was shown two years back, with AucArea ("Put Your Money Where Your Mouth Is: Evaluating Strategic Planning and Execution of LLM Agents in an Auction Arena"). The related work mentions this but fails to strike a clear difference with the proposed approach.  
- " We see the main contribution of this work as establishing a framework for considering LLM evidence as a proxy for human evidence in mechanism design" -- turning back to the framework, first, what is the framework, rather than scaffolding LLM API calls? That could still be very effective (as shown in this case, similar to many other cases), but does this framework (since so simple) extend to other examples of mechanism design? Or even broader, other strategic situations? Even if it does (which this paper does not provably demonstrate), what do we know about how LLMs would behave in out-of-distribution scenarios, which could be critical to establish fidelity in LLM simulations? 
"Can Language Models Serve as Text-Based World Simulators?" (https://arxiv.org/pdf/2406.06485?) raises serious concerns about such fidelity, especially with the GPT-4 model.
- In a twist, I would have hoped to learn something new about LLM's intrinsic behavior by this study. Why do they mimic human performance? An abundance of strategic behavioral examples in pretraining data? Does it learn meta-reasoning that it can convert into "action" just by reading textbooks or well-known auction theories? Would a bigger, more advanced model (e.g., reasoning model) perform differently from a non-reasoning model? Would a smaller (7B/32B) model perform similarly? I am left with no such answers.

### Questions
Please see the weaknesses.

### Soundness
3

### Presentation
2

### Contribution
1
