# Certifying Robustness of Agent Tool-Selection Under Adversarial Attacks

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 8, 2, 2

## Abstract
Large language models (LLMs) are increasingly deployed in agentic systems where they map user intents to relevant external tools to fulfill a task. A critical step in this process is tool selection, where a retriever first surfaces a top-N slate of candidate tools from a large pool, after which the LLM selects the most appropriate one to fulfill a task. This pipeline presents an underexplored attack surface where errors in selection can lead to severe outcomes like unauthorized data access or denial of service, all without modifying the agent's model or code. While existing evaluations measure task performance in benign settings, they overlook the specific vulnerabilities of the tool selection mechanism under adversarial conditions.
To address this gap, we introduce Certification of Agentic Tool Selection (CATS), the first statistical framework that formally certifies tool selection robustness. CATS models tool selection as a Bernoulli success process and evaluates it against a strong, adaptive attacker who introduces adversarial tools with misleading metadata, and are iteratively refined based on the agent's previous choices. By sampling these adversarial interactions, CATS produces a high-confidence lower bound on accuracy, formally quantifying the agent's worst-case performance. Our evaluation with CATS uncovers the severe fragility of SOTA LLM agents in tool selection. Under attacks that inject deceptively appealing tools or saturate retrieval results, the certified lower bound on accuracy drops close to zero. This represents an average performance drop of over 60\% compared to non-adversarial settings. For attacks targeting the retrieval and selection stages, the certified accuracy bound plummets to less than 20\% after just a single round of adversarial adaptation. CATS thus reveals previously unexamined security threats inherent to tool selection and provides a principled method to quantify an agent's robustness to such threats, a necessary step for the safe deployment of agentic systems.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper studies robustness of LLM agent tool selection, the two‑stage process in which a retriever surfaces a top‑N slate of tools and the agent chooses one to execute. It introduces CATS (Certification of Agentic Tool Selection), a statistical framework that treats each adversarial interaction as a Bernoulli trial and computes a high‑confidence lower bound on robust accuracy via Clopper–Pearson intervals. The attacker can inject up to k tools and adapt across R rounds using feedback from the agent’s previous choices; the refinement is modeled as a Markov process over tool metadata. Experiments use BFCL single‑tool tasks with M=300 tools, top‑N=10 retrieval (MiniLM-L6-v2), and several models as selectors (Llama‑3.1‑8B, Gemma‑3‑4B, Mistral‑7B, Phi‑4‑14B; Gemini‑2.5 Flash appears as an attacker). The paper defines five attack families (Adversarial Selection, Top‑N Saturation, Privilege Escalation, Abstention Trigger, Intent Shifting) and reports that the certified bound collapses toward zero under strong adaptive attacks (e.g., R=10), even when clean accuracy is high. A causal ablation shows low robustness (<0.5) even with forced inclusion of the correct tool in the slate, implicating both retriever and selector (Figure 3, p.8). The work argues CATS is the first formal statistical certification tailored to discrete tool selection rather than continuous perturbations or output text.

### Strengths
- First certification framework tailored to discrete tool selection with iterative adaptive attacks; clean formalization of pipeline and adversary space.
- Comprehensive experiments across models and attacks; informative ablations on rounds, budgets, retrievers, and frameworks
- Clear motivation (Figure 1, p.2), self‑contained algorithms (App. A.2), and compact visual summaries (Figure 2, p.8; Figures 4–8, pp.18–20).
- Reveals large robustness gaps between clean and certified performance; provides a reusable evaluation harness that can guide defense development for agentic systems.

### Weaknesses
- The lower bound is computed against the authors’ ∆adv class (templated, LLM‑generated, Markovian refinement). Claims of “worst‑case performance” should be qualified; results do not imply bounds over all possible adversaries or non‑Markov strategies. Provide an explicit statement of scope in §3.6
- Retrieval uses a single embedding model with Top‑N=10; more realistic settings include Unicode normalization, near‑duplicate clustering, homoglyph canonicalization, and slate diversification/quotas. Including at least one defended retriever baseline would strengthen conclusions about systemic vulnerability.
- The paper assumes a privilege field and compares to a user budget, but the user privilege model and enforcement are not specified; clarify how πuser is set and judged in experiments
- Results are on BFCL single‑tool calls with synthetic narrative context. It would be valuable to test on real tool stores or MCP/OpenAPI‑derived corpora and to vary M and N systematically to show scaling trends.

### Questions
- How generalizable is this certification to other adversarial strategies (e.g., non-Markov or non-templated attacks)? Could you explicitly clarify this scope in §3.6?
- The experiments use a single embedding-based retriever with Top-N = 10. Have you considered evaluating more realistic retrieval settings, such as Unicode normalization, near-duplicate clustering, homoglyph canonicalization, or slate diversification, to simulate defended retrievers? Including one defended retriever baseline could strengthen the claim of systemic vulnerability.
- In the Privilege Escalation attack, the paper assumes a privilege field π(t) and compares it to a user privilege πuser. How is πuser defined and enforced in your experiments? Are privilege mismatches detected through metadata rules or simulated policy constraints?
- All evaluations are conducted on BFCL single-tool tasks with synthetic narrative context. Have you tested or do you plan to test CATS on real tool stores (e.g., MCP/OpenAPI-derived corpora) or vary M and N systematically to analyze scaling behavior and generalizability?

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
3

### Summary
The authors study the setting of adversarially attacking the tool-selection process for agentic LLMs. They devise a small collection of prompts which they use to get a model to generate attacks. The attacks focus on two areas: (1) the slate selection phase (choosing which tools to consider in an initial narrowing-down phase), and (2) tool selection (choosing which tool to use from the narrowed-down slate). They find that models are susceptible to both families of attack.

### Strengths
## originality
As far as I know, this is the first work to explicitly study the threat model of attacking the tool pool.

## quality
The paper is well-written and the experiments are compelling.

## clarity
The paper is well written and overall quite clear.

## significance
It's not obvious exactly how important a threat model this is, given that it assumes no steps are taken to curate or moderate the tool pool. However, it is still interesting and a worthwhile addition to the literature.

### Weaknesses
It would be nice if the authors could explain a bit more clearly why their threat model is realistic. Or, if it's not realistic, to acknowledge it or explain how it could still be a problem in specific situations. My default intuition is that most of these issues go away if there is moderation of the tool pool: if it's a private company, then all their tools will be internal; if it's a public setting, then I would assume there would be some maintainers as in open-source projects who would check for this kind of malicious tool. Would tool certification/curation just solve this problem?

I would also like more discussion of what other defenses would look like, beyond tool certification/curation.

It would be good if the authors could be more clear in the paper (top and middle of page 6) that their attacks are in fact simply different prompts given to an LLM as input (which I later understood by looking at the Appendix). It was not clear just from reading that page.

### Questions
## Questions

137: I would really like citations supporting the idea that these tools can be authored by anyone. This is an important point for your paper, since the importance of the setting hinges on this being true.
202: what is semantic manipulation? Could you please explain it?
247: regarding the adaptive update, how much better is this than a simple best-of-N attack approach?
291: are you allowing k >= N? It seems like you're not, but this should be made very clear.
313: how do you choose $r$ and $k$? These seem pretty important.
349: "stability" is a weird thing to say here. It's just "to reduce noise".
360: in the multi-turn setting, what is k?
375: for top-N Saturation, is k < N? It seems that it should be. However, "saturation" and "flooding" give the image of all the N tools in the slate being malicious. Maybe you can change the language here to make it more clear.
Figure 2: what happened to the gemma3 orange bars in the first two plots?
Figure 3: can you also simply report what proportion of the time the correct tool wasn't in the slate? that seems like a much simpler way of answering this question, and you've already done the experiments for it.

## Suggestions

line 26: "severe" feels a bit strong
line 48: please provide a justifying citation to support the notion that anyone can publish malicious tools.
line 65: citet -> citep
line 68: citet -> citep
87: agentic systems -> agentic tool-calling systems
138: same as previous comment
189: consider putting a \quad after the comma before the t
191: excludes -> contains no
192: misleading -> wrong
256: I don't really understand this sentence
top of page 6: I didn't understand what exactly these attacks were until I looked at the appendix
301: at this point, it wasn't obvious to me how Privilege Escalation is different from Adversarial Selection. I think explaining clearly that these are all just different prompts would help a lot.
354: the Gemma3 citation is messed up
379: "attacker model" this is literally the first time I understood there was an attacking model. Please make it clear earlier
Figure 2: please make all plots have the same x axis order
Figure 3: this is a table, not a figure
436: while -> While

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents CATS, a statistical framework for verifying the robustness of agent tool selection. CATS models tool selection as a Bernoulli process. By simulating multiple rounds of adaptive attacks (where attackers can iteratively optimize malicious tools based on the agent's historical selection), it generates a high-confidence lower limit of accuracy, thereby quantifying the agent's performance in the worst scenario. Experiments show that under multiple rounds of attacks, the authentication robustness of multiple SOTA LLM agents drops sharply to nearly zero, revealing serious security threats in the tool selection process.

### Strengths
+ The issue is of practical significance: With the widespread application of LLM agents in tool invocation, the robustness of tool selection is indeed a critical and understudied security problem.
+ The experimental system is rich: multiple models and various attack types were evaluated, and the vulnerabilities of the retrieval device and the selector were deeply analyzed through ablation experiments.

### Weaknesses
- The key points of the paper and the content of the method section are out of balance: The core content of this paper is the robustness evaluation framework, but the methods section devotes a considerable amount of space to detailing the classification and implementation of attack methods (such as Top-N Saturation, Abstention Trigger, etc.). This leads readers to feel confused when understanding the core mechanisms of the certification framework itself (such as the composition of multiple rounds of experiments, the definition of the Bernoulli process, and the calculation of confidence intervals), and they are not clear about the priorities.
- Key method details are missing: Although the paper presents various attack types, it does not elaborate on how these attacks are specifically implemented in the system. For example, how are the three attack types such as Top-N Saturation and Abstention Trigger implemented?
- The lack of a clear threat model: The paper does not explicitly define the attacker's specific capabilities, knowledge boundaries, and restrictive conditions (for example, whether the attacker can access the retrieval device, whether they can modify existing tools, etc.), which casts doubt on the rationality and universality of the attack scenario.

### Questions
- What is the core mechanism of the certification framework?
- What are the specific Settings of the threat model?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper studies adversarial attack on the tool-selection stage of agentic systems. Specifically, the attacker could inject malicious tools and mislead agents to select them.  It formalizes robustness as the probability that the agent still picks an intent-satisfying tool even when an adaptive adversary can inject up to $k$ malicious tools and iteratively refine them over $R$ rounds. The proposed framework, CATS (Certification of Agentic Tool Selection), treats each full multi-round interaction as a Bernoulli trial and uses Clopper–Pearson intervals to produce a high-confidence lower bound on “robust accuracy.” In particular, to study the worst case setting, the paper introduces an adaptive attacker that can dynamiclly refine its attacking policy based on the agent's previus choices. Experiment results show that under multi-round attacks the certified lower bound can collapse to near zero, and even with forced inclusion of the correct tool, certified robustness stays <50%, indicating both retrieval and selection are vulnerable

### Strengths
1. Studying adversarial attack on the tool-selection stage of agentic systems is well motivated.

2. Formalizing a multi-round problem instead of a single round is more realistic, which unlocks the potential to study advanced red team strategies (e.g. adaptive attacks studied in this paper).

3. Results are evaluated across multiple attack strategies (Top-N Saturation, Adversarial Selection, Privilege Escalation). The near-zero certified lower bound convincingly show that this is a real problem.

### Weaknesses
1. overclaim on novelty. As far as I know, this is not the **first** paper studies adversarial attack on the tool-selection stage. see https://arxiv.org/pdf/2412.10198 and https://arxiv.org/pdf/2508.02110v1. 

2. Lack of experimenting with more defense methods. For example. how easy it is to catch these injected malicious tools? Can the blue team easiy select them with an additional monitor before using the retriever and selector?

3. Studying the worst-case setting of adaptive attacks is reasonable. However, in practice, the attacker might not really have access to the agents' detailed trajectories since they are often hided by companies?

### Questions
1. What was n (number of trials) per model/attack in practice? How sensitive were your lower bounds to halving n? A small table of “trials → CI width” would make the “certified” claim more concrete.

2. The current paper tests defender LLaMA-3.1 vs. attacker Gemma-3 as a “representative” strong attacker (P7). Did you try mismatched or weaker attackers? Do we still see near-zero bounds when the attacker LLM is strictly smaller or older than the defender?

3. The current paper focuses on single-tool tasks. How does the proposed attack adapt to multi-tool tasks?

### Soundness
2

### Presentation
2

### Contribution
2
