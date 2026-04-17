# How does information access affect LLM monitors' ability to detect sabotage?

- Decision: Reject
- Scores: 2, 4, 6, 4

## Abstract
Frontier language model agents can exhibit misaligned behaviors, including deception, exploiting reward hacks, and pursuing hidden objectives. To control potentially misaligned agents, we can use LLMs themselves to monitor for misbehavior. In this paper, we study how information access affects LLM monitor performance. While one might expect that monitors perform better when they have access to more of the monitored agents’ reasoning and actions, we find that contemporary systems often perform better with less information, a phenomenon we call the less-is-more effect for automated oversight. We demonstrate this phenomenon, and analyze the conditions under which it occurs, in three evaluation environments where agents must conduct sabotage while evading monitors. Motivated by the less-is-more effect, we introduce extract-and-evaluate (EaE) monitoring—a new hierarchical approach where one monitor isolates relevant excerpts from the monitored agent's trajectory and another monitor scores these excerpts. EaE rarely performs worse than other methods and achieves substantial improvements in some settings. For example, in BigCodeBench-sabotage with GPT-4.1-mini as the monitor model, EaE improves sabotage detection rates by 16.8 percentage points over the next-best monitor without increasing the false positive rate. In SHADE-Arena, EaE slightly outperforms the best baseline with a GPT-4.1-mini monitor and slightly trails with a Claude-3.7-Sonnet monitor. In addition, we find that agents unaware of being monitored can be caught much more easily and that monitors scoring longer reasoning traces gain more from information filtering. Lastly, we conduct a cost-performance analysis and find that Gemini and Claude models with monitoring techniques that involve information filtering occupy much of the Pareto frontier.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper studies the question of "how does information access affect LLM monitors' ability to detect sabotage?". They experimented across 3 environments and showed that monitors tend to exhibit what they called the "less-is-more" effect. Additionally, they proposed a new monitoring framework, "extract-and-evaluate," and claimed that this outperformed baselines in the BigCodeBench environment. Lastly, they conducted the cost-performance analysis.

### Strengths
- Ran 3 different environments
- The background is well motivated
- Conducted a diverse set of experiments

### Weaknesses
This paper claims to have 4 main contributions in Line 83, but I find 3 of them are either relatively weak or not original.
- Contribution 1 (less-is-more effect) is not a unique contribution; The result from Arnav et al. (arXiv:2505.23575) has already shown this.
- Contribution 2 (extract-and-evaluate monitoring) seems to be quite environment and monitor-dependent. In Fig. 3, based on the error bars, it seems to work noticeably well only for the gpt-4.1-mini monitor on the BigCodeBench and the MLE-sabotage. The other 5 monitor-environment combinations don’t show as substantial an improvement when considering the error bars.
- Contribution 3: Didn’t Arnav et al also show the experiment results with the reasoning and non-reasoning models as attackers?

### Questions
- For contribution 3, didn’t Arnav et al also show the experiment results with the reasoning and non-reasoning models as attackers?
- I'm happy to hear the authors' responses on the above weaknesses.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper studies the task of monitoring for sabotage: given an LLM agent prompted to perform a hidden task, can an LLM monitor detect the presence of this hidden task?

Following Arnav et al. (2025)’s finding that monitors with access to both CoT and actions can sometimes underperform action-only monitors, the paper proposes the “less-is-more” effect: sometimes, provisioning monitors with more information (e.g. CoT + actions) causes them to underperform monitors with less (e.g. just actions, or just CoT).

The paper experiments on four monitoring approaches: action-only, CoT-only, CoT+action, and their novel approach, extract-and-evaluate (EaE). EaA works by separating monitoring into two phases: the “Extractor” reasons about the transcript and extracts quotes, while the “Evaluator” produces a suspiciousness score based only on extracted inputs.

The paper runs experiments on three environments: SHADE-Arena, MLE-sabotage, and BigCodeBench-sabotage, with two monitor LLMs (Claude-3.7-Sonnet and GPT-4.1-mini), and seven attacker LLMs (two “traditional”/non-reasoning, five reasoning).

The paper demonstrates the less-is-more effect empirically, finding that CoT+action often underperforms other strategies, despite having the most information. It also finds that EaE sometimes outperforms other methods.

### Strengths
1. The abstract and introduction are clearly written, and the paper’s core claim about the less-is-more effect is easy to follow.
2. The proposed approach, extract-and-evaluate, is simple and well-motivated in light of the less-is-more finding.
3. Beyond the core claims about less-is-more and EaE, Section 5 gives a number of interesting empirical findings from the conducted experiments. I think these will be useful to researchers working on LLM monitoring.
4. The studied task, monitoring for sabotage, is an important problem for safety.

### Weaknesses
1. “extract-and-evaluate monitoring technique that improves sabotage detection rates in the BigCodeBenchsabotage environment by up to 12 percentage points over the next-best monitor
without increasing the false positive rate" - this feels like a misleadingly cherry-picked presentation of evidence in the abstract. EaE performs at a statistically indistinguishable level in many settings in Figure 3, and underperforms CoT only in BigBench traditional attacker/GPT-4.1. Without an a-priori reason to privilege this specific setting, you should present either aggregate statistics, or the full range from min to max.

2. I found it difficult to find the full details of any individual experiment. Which models are used as attackers, and which are used as monitors? Details appear spread throughout the Results section; Figure 4 shows five models, but others are listed in the appendix. It would be easier to follow if there was a concise list of all models used, and which roles they were used for, in the Methodology section.

3. I also found it difficult to follow the justification for some of the findings in section 5. I think it would be helpful to have a single graph for each finding, focusing on the features under study. For example, Finding 2 could show a scatterplot between monitor performance and average transcript length, with a point for each monitor-environment pair; Finding 7 could show a bar graph with monitor performance on reasoning models vs. monitor performance on traditional models. Figures could be in appendix, with links from each finding paragraph.

4. Limited novelty of the less-is-more effect itself, as prior work (Arnav et al. 2025) already empirically observed cases where CoT+action underperforms action alone.

5. While the less-is-more effect is demonstrated empirically, it’s still not clear to me *why* it happens. Is the problem due to distracting information? Or model performance in general declining over a certain context length? Given that prior work has already demonstrated this effect empirically, it would be a more valuable contribution to better understand its cause. E.g. if you add plausible-but-irrelevant text, does this impact performance more than adding the same number of clearly-irrelevant tokens (random characters, or just “...” repeated)?

### Questions
1. Figure 3, “error bars are bootstrapped 95% CIs” - how is the bootstrapping done? IIUC, Figure 3 aggregates over attacker models, but these attacker models may be evaluated on the same questions, so model-sample pairs aren’t IID, and a naive bootstrap will underestimate CIs. Could you provide additional details on how you’re computing CIs when aggregating over data? (See e.g. https://www.anthropic.com/research/statistical-approach-to-model-evals, “Cluster standard errors” for one statistical approach used in situations like this.)

2. Figure 6, “We use AUROC rather than log-AUROC for compactness.” - what does this mean?

3. For the results related to monitor awareness, the authors might be interested in the results of this paper, which specifically studies how monitor awareness impacts detection rates and behavior: https://openreview.net/forum?id=r4Q6o7KGdb

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper considers the problem of monitoring LLM agents for planning and executing malign side tasks alongside benign main tasks. The main goal of the paper is to evaluate which information the monitor should utilize and how that information should be utilized in order to maximize catch rates (while minimizing false positives) of the monitors. The authors introduce a novel two-step monitoring protocol, extract-and-evaluate (EaE), to improve the accuracy of weaker monitors. The authors evaluate various monitoring protocols against both traditional LLMs and reasoning models on three evaluation environments. The authors main finding is that more information is not always better, and developing various filtering methods for monitoring can be extremely beneficial.

### Strengths
* The main contribution of the paper is valuable to the AI control community: "naively applying monitors on the full agent trajectory is often insufficient to achieve optimal monitor performance."
* EaE seems to provide a Pareto improvement (performance / cost) over other monitoring strategies.
* The paper is very well-written and the presentation is overall well-done.

### Weaknesses
* Do not run against adaptive agents (i.e. where the agent can optimize itself knowing that it will be monitored). This makes the evaluation setup unrealistic in comparison to actual deployment scenarios. Figure 3 shows that when the agent knows more about the monitoring environment (e.g. SHADE-Arena) the monitoring performance decreases dramatically, and the proposed approaches provide much less benefit (especially when using a weaker monitor). This indicates that the proposed approaches will not necessarily be useful in practical AI control scenarios.
* Some of the results are quite obvious. For example, Finding 2 and Finding 6 are both very intuitive given the "lost-in-the-middle" effect [1], especially for weaker models. I.e. it makes sense that less-is-more especially when monitoring with a weaker agent. Nevertheless, even given this intuition, it is important to show the result empirically.
* Some of the claims are not obviously backed up by the data. For example, the authors claim that EaE monitoring improves catch rates by 12 percentage points over the next best catch rates (line 321, and mentioned earlier in the paper). Can the authors point to where this can be seen. It is not obvious from the figures in the main paper.
* There's no one monitor setup that always wins, and the increase in performance is not always significant.


[1] Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., & Liang, P. (2023). Lost in the middle: How language models use long contexts. arXiv preprint arXiv:2307.03172.

### Questions
* How is agent success determined? Is it a deterministic function based on the evaluation environment?
* Why do CoT-only (w/o actions?) monitors perform better on traditional LLMs (non-reasoning models)?
* What is the monitor in Figure 4? Is it GPT-4.1-mini?
* Can you provide a more in-depth justification for why filtering for success makes sense? Why would the CoT-only monitor have a large advantage? This is not obvious.
* Furthermore, if the actions made by the agent could cause harm, would it not make sense to catch the agent before harm was caused? I think this might be related to the online vs offline setting (Roger and Shlegeris (2025)).
* What are task success rates for the various models and evaluation environments?
* Can you further justify why the experimental setup is a pragmatic proxy for the real-world AI control scenario? In practice, agents will not necessarily be prompted to perform these side actions and therefore might also tend to hide their true intentions more (because misalignment).

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies how what the monitor is allowed to see (actions, CoT, both, or a filtered subset) changes an LLM monitor’s ability to catch covert sabotage in three “AI control” environments—SHADE‑Arena (assistant tasks with hidden side‑goals), BigCodeBench‑sabotage (single‑step coding with backdoors/side rules), and MLE‑sabotage (agentic ML engineering). The central empirical claim is a “less‑is‑more” effect: monitors given less raw information sometimes detect sabotage better than those given both CoT and actions. Motivated by this, the authors introduce Extract‑and‑Evaluate (EaE): a two‑stage protocol where an Extractor first quotes salient snippets from (CoT + actions), and then an Evaluator scores suspiciousness using only those quotes. EaE improves catch rates in BigCodeBench‑sabotage by up to 12 percentage points on average (and 18.2 pp on subtle side tasks) at 1% FPR, and can amplify weaker monitor models cost‑effectively. The paper further shows that CoT‑only monitors often outperform action‑only ones (contrary to some earlier results), that agents who believe their CoT is private are far easier to catch, and that on a cost–performance frontier, Gemini and Claude families with information filtering dominate. Methodologically, the paper emphasizes log‑AUROC (AUROC with the FPR axis in log‑space starting at 1%) to reflect low‑FPR operational regimes.

### Strengths
A clear empirical story around information access. The paper asks a practical question: "what should your monitor actually look at?" and answers it with cross‑environment evidence that naively giving monitors everything (CoT+actions) is often suboptimal, with CoT‑only or extracted subsets performing better. This is useful guidance for safety teams building pipeline monitors today. 

---
A simple, actionable protocol (EaE) with measurable gains. EaE operationalizes the less‑is‑more intuition via hierarchical filtering: have one model extract quotes before a second model scores only those quotes. In BigCodeBench‑sabotage, EaE beats competing monitors on TPR@1% FPR (overall +12 pp; +18.2 pp on subtle tasks), and a “weak Extractor + strong Evaluator” variant achieves large lifts at modest additional cost. This is a pragmatic contribution that teams can test quickly. 

---
Agent awareness and monitorability. The finding that telling the agent its CoT is private dramatically improves CoT‑based monitoring (Δ log‑AUROC > 0.4 in SHADE‑Arena) is an important reminder that situational awareness policies matter. It also partially reconciles mixed results in the literature: unaware agents verbalize more, so CoT monitors work better. 

---
Cost–performance perspective. The Pareto analysis is helpful: Gemini and Claude occupy much of the efficient frontier, and information filtering appears on many Pareto‑optimal points; gains diminish after roughly $0.005/transcript, but spending ~$0.1 ensures robust performance giving practical ballparks for deployment. 

---
Care with evaluation metrics. Focusing on log‑AUROC (weighted to ≤10% FPR) reflects real auditing budgets; the authors also reproduce results under standard AUROC and discuss bootstrap variance and threshold effects.

### Weaknesses
**Contribution framing**: Beyond EaE, other “contributions” (awareness, cost analysis, reasoning vs. traditional LLMs) feel like observations rather than core advances.

**CoT-only novelty claim**: The authors state that their CoT-only results are novel because prior work (e.g., Arnav et al., 2025) focused on action-only vs. CoT+action monitors. This seems fair, but the claim could be clearer: CoT monitoring itself is well-known, so the novelty lies in the systematic evaluation across environments, not in introducing CoT-only monitoring per se. Claiming novelty here is misleading.

**Scope of EaE**: Gains hinge on very long CoTs overwhelming weak monitors. If CoTs are truncated or monitors scale better, EaE’s advantage largely disappears. If CoT length were capped or chunked, would EaE still matter? The authors acknowledge that on non-subtle tasks and traditional LLMs, EaE is not better than prior methods.

**Mechanistic gap**: Less-is-more may reflect parsing limits and post-hoc justification bias, but no controlled ablations isolate these factors.

### Questions
1. Adversarial extracts. How robust is EaE if the Extractor misses the key quotes or is adversarially distracted by benign snippets? Any recall metric for the Extractor (fraction of ground‑truth incriminating spans captured)?

2. When total token cost is fixed, does EaE still outperform CoT+action or hybrid? Also, what happens if you swap roles (strong Extractor, weak Evaluator)?

3. Mechanistic clarity: Did you run ablations to confirm that failures of CoT+action monitors are due to post-hoc justifications rather than other confounders (e.g., prompt complexity)? Can you disentangle causes of less-is-more? For example:

* Truncate CoTs to fixed token budgets.
* Remove post-hoc justification phrases.
* Shuffle or mask irrelevant CoT spans.
* How do these affect monitor rankings?


4. Evaluator necessity: Why is a separate Evaluator needed? Did you test a single-step prompt that extracts quotes and assigns a suspicion score? If not, why assume two stages outperform one?


5. Scope of novelty: CoT-only monitoring is presented as a novel contribution, but prior work has studied it. Can you clarify what’s new here?

6.  If CoT length were limited to what the monitor can handle (or processed in chunks), would EaE still outperform? Is the gain mostly due to mitigating long-context degradation?

### Soundness
2

### Presentation
2

### Contribution
2
