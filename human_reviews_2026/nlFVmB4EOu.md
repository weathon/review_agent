# Semantic Anchoring in LLMs: Thresholds, Transfer, and Geometric Correlates

- Avg Score: 2.67
- Decision: Reject
- Scores: 2, 4, 2

## Abstract
We propose *semantic anchoring*, a unified account of how LLMs turn pretrained capacity into goal-directed behavior: external structure binds latent patterns to a target. Unified Contextual Control Theory (UCCT) formalizes this with an anchoring strength $S=\rho_d-d_r-\log k$, where $\rho_d$ is within-target cohesion, $d_r$ is prior–target mismatch, and $k$ is the anchor budget. UCCT anticipates threshold-like flips and \emph{strictly generalizes} in-context learning, while reading retrieval-augmented generation and light fine-tuning as anchoring variants. Evidence comes from three studies: E1 shows cross-domain anchoring in text and vision where coherent anchors rebind strong priors; E2 varies representational familiarity via numeral bases at fixed computational complexity and observes ordered few-shot thresholds, transition widths, and transfer trade-offs that track $\rho_d/d_r$ and $S$; E3 analyzes layer-wise geometry and finds a geometry-to-behavior *correlate*, with peak anchoring and normalized area summarizing trajectories and correlating with internal shot midpoints. UCCT offers a testable lens and practical proxies for prompts, retrieval, and light tuning, while motivating targeted causal probes.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The manuscript tries to unify the behavior change of LLMs (i.e. overwriting pre-training behavior) induced by any kind of modifications, including in-context learning, fine-tuning, etc. The authors proposed a UCCT framework. UCCT claims three additive factors contributing to the LLM behavior change (Eq.2). The three factors are 1) the strength of the overwriting samples, 2) the distance between new samples and pretraining samples, 3) number of new samples. The authors support UCCT framework by 3 experiments (E1-3).

UCCT conceptually makes sense. However, the overall quality of the manuscript is poor. See weakness below.

### Strengths
The authors focused on a question that how LLMs can adapt to new rules that contradict pre-trained rules. The proposed framework tries to unify such an ability across in-context learning and fine-tuning, and across modalities. The unified view is novel, though it lacks rigorousness and justification.

### Weaknesses
**Vague jargon and terms**

Many terms are very vague without clear definitions. It is difficult to grasp what the study is about, because some common concepts are referred to in unconventional ways. In the introduction, “goal-directed behavior”, “latent pattern”, “anchor”, “target” lacks clear definition.  The authors give some examples of “anchors” in line 167, “few-shot examples, retrieved text, instructions”. Is an anchor just a “condition”? Line 172, "P_T is a latent target cluster" is confusing. It is a cluster of what? And what is a representation of a cluster? In Line 189, "ρ_d(P_T ) is within-cluster density", it is confusing what it means.

**Lack of rigorousness in the formulation**

In line 54-62, the authors present a UCCT theory. However, the formulation lacks rigorousness where symbols are described in ambiguous words. There is a major issue that Eq (1) is wrong and misinterpreted. It could be a typo, the factor p(y|P_T,A) on the right side should be p(y|P_T,A,C). The Bayesian interpretation is wrong. Eq (1) does not contain any Bayesian view. p(P_T|A,C) is not a posterior, but more like the prior. A posterior of P_T would be p(P_T|y,A,C).

**The "threshold" claim is hypothetical**

There is insufficient support for the claim of “threshold” behavior. First of all, it is hard to believe there is a sharp behavior transition while continuously changing a condition. It could be a sharp transition if some measure is computed using T=0 greedy sampling, but more generally the logit probability should shift continuously. Second, given the definition of S, there are 3 ways to drive it across a “threshold” given three terms. A valid support experiment to show there is a genuine threshold would be regardless of the ways, there always a fixed value of S when the behavior changes. 

**Missing sufficient background**

Only few related papers are cited. Line 88, the authors discussed “previous work” without citing any paper. Generally, the manuscript does not give a good overview on what has been studied in ICL, and pre- and post-training knowledge/skills, which is studied in numerous papers in the past few years.

**Figure quality and disorganized structure**

Figure 1a and Figure 8 are duplicated, and it is very difficult to get the key information in the figure. The symbols and arrows in the figure look random placed.

The structure of section 4 is disorganized. With too many sub- and subsub- sections, it is hard to follow the exact experiment design and results.

### Questions
See weakness, the formulation of the problem is poor. Each symbol should be clearly defined in a formal way rather than described in words.  For example, "latent target cluster", each of the three words needs to be explained. 1) What is a target? 2) cluster of what and how to get them 3) why is it latent.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes **semantic anchoring**, a novel view for explaining how external sources bind the learned latent patterns to a target pattern cluster.
The core contribution is the **Unified Contextual Control Theory (UCCT)**, which formulates the difficulty of binding latent patterns to target ones by a single equation: $S=\rho_d-d_r-\log k$. Based on my understanding, $\rho_d$ denotes the target distribution, $d_r$ is the distribution distance between the target pattern and lattent pattern distributions, $\log k$ is a term representing the effect level of target domains, i.e., with more samples from the target domain, the effect should be larger.
The authors conduct various experiments to rationalize their design choice.

### Strengths
1. The semantic anchoring view is novel and interesting.
2. Many experiments to show the robutsness of the proposed UCCT criteria.

### Weaknesses
First of all, I need to claim I am not an expert in this domain. But I feel this paper is more like a free oral presentation rather than an academic paper. **The organization is a bit mess, with no clear definition before presenting a concept.** For example, I have no idea what `E1, E2, E3` are in the first three sections until I read L258-L263. And I cannot find a clear definition of `B10, B8, B9` in Eq.(4). Such oral-style writing makes this paper extremely hard to follow.

Secondly, I have a basic doubt regarding the `semantic-anchoring assumption` (L158-L160). Since the anchors "activate" target pattern clusters $P_{T}$, it should only work when the pretrained LLM has already seen this pattern. **But this assumption does not work for many ICL/RAG/finetuning scenarios, as we are actually injecting new concepts (patterns) that the LLM has not seen before.** Under such circumstances, there is no existing latent patterns to activate or to bind, so the semantic-anchoring assumption does not stand.

Thirdly, I do not agree with the statement *"Richer interactions (e.g., multiplicative terms) are plausible but would add parameters and reduce identifiability. Empirically, the linear form is sufficient"* (L207-L209). **Because the $\log k$ should be correlated with $\rho_d(P_T)$** if I understand correctly. It is exactly the fed $k$ in-context samples that shape the target distribution $P_T$, therefore, increasing $k$ should also increase $\rho_d(P_T)$ and decrease $d_r(P_{prior},P_T)$ seems to be a more reasonable explanation for Eq.(2). But I still think we should bind $k$ with $\rho_d(P_T)$ instead of isolating them in a linear equation.

Finally, I think this criteria is not easy to adopt in practice, as we cannot obtain the embeddings from many API-called models like Gemini and Claude and conduct such analysis.

### Questions
Can you elaborate more about the details of API evaluations (M1-M4) since it seems we cannot get the embeddings w.r.t. specific layers for these models?

### Soundness
3

### Presentation
1

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
This paper studies the theory explaining the behaviors of in-context learning of LLMs. The authors propose Unified Contextual Control Theory (UCCT), a semantic anchoring lens for how LLMs convert pretrained latent patterns into goal-directed behavior. UCCT adopts Bayesian theory to predict the threshold-like flips in behavior across prompting, retrieval, and light fine-tuning of LLMs. The authors also conduct three experiments to verify UCCT, including cross-domain anchoring, numeral-based arithmetic calculation and layer-wise geometry analysis.

### Strengths
1. The UCCT theory offers a new explanation that subsumes ICL, RAG, and light fine-tuning as instances of a single anchoring process governed by S.

2. The authors present UCCT with several practical and measurable proxies, including thresholds, transition widths, and transfer trade-offs for analysis in practice.

3. Some experiments verify the theory of UCCT.

### Weaknesses
1. Presentation is poor and hard to follow:
- The figure size and scale are hard to read;
- There are multiple undefined nouns in figures, like "commensense-0, commensense-1, commensense-2" etc., as well as in text like "$P^{(B)}_T$;
- Lots of texts do not convey clear meanings "We instantiate three numeral systems with differing expected density under pretraining exposure: Base 10 (higher), Base 8 (moderate), Base 9 (lower). A lightweight audit over public web/code corpora finds decimal ≫ octal > nonary; embedding-based ρ d yields 10 > 8 ≈ 9, so we expect lower k 50 for base 10 and similar k 50 for 8/9."

2. It's unclear why one would prefer UCCT, instead of other explanations and theories as discussed in the related work.


3. Indeed, it seems there is no evidence verifying the key assumptions that "LLMs are latent pattern repositories", or some other **implicit** assumptions like "Additivity in (2) assumes the three contributors act approximately independently at the margin; this gives an interpretable, parsimonious log-odds style surrogate. R". 

4. Given that, the formulation of the theory seems to be used for interpretation, instead of providing meaningful insights. The takeaway messages from the UCCT are already formulated once the assumptions are constructed. 

5. Experiments are only focused on simple and synthetic examples.

6. Given the simple theoretical formulation and experiments, the technical novely of this work is also limited.

### Questions
Please find my questions in the section above.

### Soundness
2

### Presentation
1

### Contribution
2
