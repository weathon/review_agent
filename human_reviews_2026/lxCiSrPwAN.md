# Turning Bias into Bugs: Bandit-Guided Style Manipulation Attacks on LLM Judges

- Avg Score: 4.67
- Decision: Reject
- Scores: 2, 4, 8

## Abstract
Large Language Models (LLMs) are increasingly employed as automated judges for evaluating generative models. 
However, their known *stylistic biases*, such as a preference for verbosity or specific sentence structures, present an underexplored *security vulnerability*. 
In this work, we introduce **BITE** (**BI**as explora**T**ion and **E**xploitation), a black-box adversarial framework that learns semantics-preserving edits to mislead the judgment and **artificially** inflate judged scores. 
We cast the selection of stylistic edits as a contextual bandit problem and use a LinUCB policy to adaptively choose edits that maximize the judge’s score without access to model parameters or gradients. 
Theoretically, we prove a formal regret guarantee for our BITE, demonstrating its ability to efficiently learn to manipulate a judge in the realistic setting of model misspecification.
Empirically, we test BITE across a diverse range of LLM judges and tasks, including both pointwise and pairwise comparisons on chatbot leaderboards and AI-reviewer benchmarks. BITE achieves an attack success rate \(>\!65\%\) and raises scores by \(+1\)–\(2\) on a 9-point scale, while maintaining semantic equivalence. 
We further uncover model-specific "vulnerability fingerprints": judges differ in sensitivity to sentiment, register, and structural cues (e.g., headers), limiting cross-model transferability. Finally, we evaluate stealthiness and show that BITE evades standard style-control and simple detection baselines.
Our findings expose a fundamental weakness in the LLM-as-a-judge paradigm and motivate robust, attack-aware evaluation, e.g., style normalization, randomized prompting, and adversarial training of judges.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This work proposes BIas exploraTion and Exploitation (BITE), a black-box adversarial attack that inflates LLM-as-a-judge scores. Authors gather a set of style modifications that have been shown in the literature to affect scores of LLM judges. Given this set of style modifications, authors frame the problem as a contextual bandit problem, where for a given representation of the query and answer to evaluate, one needs to choose the style modification that will increase the score of the answer the most. Authors employ LinUCB to solve the problem and show practical improvements over random and rewriting baselines. In their experimental evaluations, authors show that different LLMs are vulnerable to different transformations. Additionally, Linear Style Control defenses are shown to be ineffective.

### Strengths
- The motivation and significance of the work is good: The attack is motivated by the existing vulnerabilities of LLM judges to different style modifications.  The casting of the problem as a contextual bandit problem is also clear and well motivated. Authors also display the significance of studying attacks on LLM judges as they are widely used to evaluate the performance of LLMs.

- Interesting experimental insights: Authors show that different LLMs have a particular vulnerability fingerprint and attacks do not generally transfer from model to model. Additionally, authors show effectiveness of their attack in relevant tasks such as automatic peer review.

### Weaknesses
- Poor comparison with state-of-the-art: While authors focus on the more realistic black-box setup, 3/5 models evaluated in this work are open-source, allowing the use of white-box attacks like JudgeDeceiver [1]. A comparison against attacks of this kind would allow a better assessment of the strength of the proposed attack. 

- Poor defenses: Authors analyze the score correction proposed in [2]. However, authors didn’t consider simpler defense strategies. Similarly to how their attack consists of an LLM applying a style modification strategy, an LLM could be prompted to remove style from an answer before being fed to the LLM judge. Similarly, the prompt in Figure 9, does not contain any instructions to ignore sentiment, authority, verbosity… or any other biases considered in the attack.

- Unclear motivation/need for theoretical results: In Theorem 5.1 authors build on the theory of [3] to prove a regret bound for the LinUCB algorithm on their setup. However, it’s unclear why this is needed or how this theoretical result is novel or interesting. 

**References**

[1] Shi et al., Optimization-based Prompt Injection Attack to LLM-as-a-Judge, CCS 2024

[2] Li et al., Does style matter? Disentangling style and substance in Chatbot Arena, https://lmsys.org/blog/2024-08-28-style-control/ 2024

[3] Abbasi-Yadkori et al., Improved algorithms for linear stochastic bandits, NeurIPS 2011

### Questions
- Can you employ white-box attacks over Llama3, Qwen3 and DeepSeek-R1?

- Can you evaluate other baseline defenses like rewriting the answer to remove all biases?

- How did you choose the prompt in Figure 9? Does your attack also work with other prompts used in practice, e.g., Figures 6 and 10 in [5]? 

- How novel is your theoretical result?

- How are your biases different from the ones considered by [4]?

- Can you show examples of the final attacked answers?

- The authority bias might be due to the LLM not being able to check sources. Did you test LLM judges equipped with the web search tool?

- Which prompt did you use to apply each style modification?

**References:** 

[4] Chen et al., Humans or LLMs as the Judge? A Study on Judgement Bias, EMNLP 2024

[5] Zheng et al., Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena, NeurIPS D&B 2023

### Soundness
2

### Presentation
3

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
The authors propose a new  black-box attack framework called BITE against LLM judges that formulates stylistic manipulation as a contextual bandit problem. Using LinUCB to adaptively select semantics-preserving edits from known stylistic biases, the method achieves >65% attack success rates and increases scores by 1-2 points across diverse benchmarks.

### Strengths
1. Formulating the attack as a contextual bandit problem with exploration-exploitation tradeoff is creative.  

2. Theoretical guarantees on regret bounds under model misspecification, though under a Linear model assumption so not sure if this even implies in practice for non-linear LLMs.

3. Comprehensive evaluation: Tests 5 diverse judges across chatbot benchmarks and peer review.

### Weaknesses
1. Motivation to propose a new attack formulation using  bandit formulation is unclear. 
2. Weak Baseline Comparison. Look at question section below for detail.

### Questions
1. It is unclear to me why a new bandit formulation was needed in this scenario? I see a discussion on prior work in section 2 but doesn't discuss on detail on what are the shortcomings of those approaches and why this formulation is need. Additionally, there has been search based techniques, fine-tuning attacker llm based on feedback, etc in jailbreak. Why can't those techniques be adopted here?

2. The evaluation lacks comparison with prior work. The authors mention in related work that direct attacks are generally more overt and easier to detect but the evaluation doesn't have plots to show that their proposed attack is more stealthy than prior attacks.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper reframes stylistic bias in LLM-as-a-judge systems as an active attack surface and proposes BITE (BIas exploraTion and Exploitation), a black-box, semantics-preserving style-manipulation attack driven by a contextual bandit with a LinUCB policy. BITE iteratively applies style edits (verbosity, markdown, sentiment, etc.) to a candidate answer and uses the change in the judge’s score as reward; the loop is shown in Figure 1 (p. 4) and formalized via Algorithm 1.  
The authors prove a regret bound under model misspecification (Theorem 5.1), then evaluate across pointwise and pairwise settings on multiple judges (o3-mini, Gemini-2.5-Flash, Llama-3.3-70B-Instruct, DeepSeek-R1-0528, Qwen3-235b-a22b), reporting >65% attack success and +1–2 points (on a 9-point scale) while claiming semantic equivalence. 
Experiments reveal model-specific vulnerability fingerprints (e.g., consistent bias for verbosity/italics but divergent responses to special characters) and asymmetric, generally low transferability across judges (Figure 3 & Figure 4). 
A style-control defense (linear debiasing on simple features) barely reduces inflated scores (Figure 5), and a meta-judge explanation-based detector shows high false positives on simpler tasks and very low detection on harder ones. The paper ends by arguing for attack-aware evaluation (style normalization, randomized prompting, adversarial training).

### Strengths
•	Clear threat model and attack design; black-box, query-constrained setting matches real systems (APIs, rate limits).  
	•	Principled learning method (LinUCB) with regret analysis under misspecification; the choice trades off simplicity and performance.  
	•	Strong empirical wins across pointwise/pairwise evaluations and on an AI-review case study (Table 1), with steady improvement over baselines (Figure 2).  
	•	Model-specific diagnostics: “vulnerability fingerprints” (e.g., near-universal verbosity bias; divergent reactions to special characters) and low/asymmetric transfer, giving actionable insights for defense design.  
	•	Defense evaluation: Linear style-control and explanation-based detection both underperform, underscoring depth of the issue.

### Weaknesses
•	Defense scope: Only simple linear debiasing and a meta-judge detector are tested. Consider stronger baselines (non-linear debiasing, randomized prompting at evaluation time, adversarially trained judges) to align with the paper’s own mitigation suggestions.  
	•	Bandit policy ablations: LinUCB is compared to Random and rewrite-only baselines, but not to other bandit/RL choices (e.g., Thompson Sampling, nonlinear contextual models). This would help separate gains due to exploration strategy vs. action set.  
	•	Semantic-preservation check relies on embedding similarity; might miss subtle meaning shifts caused by tone/register changes (especially with authority/sentiment edits). A small-scale human validation or factual consistency metric would strengthen claims.  
	•	Generalization beyond text QA: Most tasks are chat/QA and automated review. It’d be useful to test code-judging or multimodal judging to probe boundary conditions. (Judge prompts are provided, which is helpful.)  
	•	Practicality assumptions: Helper LLM (Gemini-1.5-Flash-8B) performs the rewrites; cost/latency trade-offs and sensitivity to helper choice aren’t deeply analyzed.

### Questions
1.	Misspecification parameter ζ(T): How is it estimated or bounded in practice when setting the UCB parameter α? Could you report empirical sensitivity to α under varying ζ(T)?  
	2.	Policy alternatives: How would Thompson Sampling or a simple nonlinear bandit (e.g., kernelized or shallow NN) perform vs. LinUCB under the same budget?
	3.	Action-set sensitivity: Which of the nine edits drive the majority of gains per judge? An ablation that removes each edit (one-out) would clarify leverage points. (Appendix A.1 lists the biases.)  
	4.	Helper LLM choice: Do results change materially with a smaller/larger helper or a different family? Any evidence of helper–judge family interactions?  
	5.	Semantic preservation: Beyond embeddings, did you try factual-consistency checks or targeted human audits on a subset, especially for authority/sentiment edits?  
	6.	Judge prompts: Since judges and prompts strongly shape outputs, how robust are results to prompt randomization (as suggested in mitigation)?  
	7.	Transfer asymmetry: You hypothesize “preference leakage” explains asymmetry (Figure 4). Any supporting evidence from data provenance or training overlaps you can include (even indirect)?  
	8.	Query-budget scaling: You fix T=25. What is the ASR curve as T varies (e.g., 5–100)? Is there early-round over-exploration that could be optimized?

### Soundness
3

### Presentation
3

### Contribution
4
