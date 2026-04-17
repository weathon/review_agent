# Reranker Helps, but Not Enough: Towards Strong Poisoning Attacks Against RAG

- Decision: Reject
- Scores: 4, 6, 2, 4

## Abstract
Retrieval-Augmented Generation (RAG) augments Large Language Models with timely, external information, making their retrieval corpora a prime target for data poisoning. 
However, existing targeted poisoning attacks exhibit limited effectiveness against RAG equipped with a reranker to enhance retrieval quality.
Remarkably, this defensive benefit comes at no additional cost: a reranker fine-tuned only on benign, in-domain documents can effectively filter malicious content without any adversarial training. 
To realistically evaluate RAG and strengthen red-teaming efforts, we conclude practical prompt design principles that reveal reranker blind spots.
Building on these insights, we introduce the $\textbf{P}$rompt-$\textbf{P}$erturbation $\textbf{P}$oisoning $\textbf{A}$ttack ($\mathbf{P}^3 \mathbf{A}$), a novel framework for generating sophisticated poisoned documents. 
$\text{P}^3\text{A}$ first employs rule-based prompt engineering to craft initial poisoned texts designed to evade reranker filtering.
It then injects subtle character-level perturbations into these texts, which promotes their ranking by the reranker while maintaining their adversarial effectiveness. 
These perturbations introduce only about 1\% textual change, ensuring the poisoned texts remain natural and readable.
Extensive experiments demonstrate that our methods achieve effective attack performance, compromising reranker-enhanced RAG pipelines.
Furthermore, our method exhibits strong transferability, proving equally effective against vanilla RAG—offering a more realistic and challenging benchmark for evaluating defense mechanisms.
Code is available in the supplementary material.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper studies data poisoning attacks on Retrieval-Augmented Generation (RAG) systems and asks whether a reranker—a common module in realistic pipelines—can mitigate these attacks. The authors first show that adding a reranker indeed reduces the attack success rate of several existing poisoning methods, implying that prior work overestimates the vulnerability of RAG. To expose the remaining weaknesses, they propose a two-stage Prompt-Perturbation Poisoning Attack (P3A) that uses (1) rule-based prompt generation and (2) character-level perturbations guided by reranker gradients.  Experiments on multiple datasets and models demonstrate that P3A restores high attack success even with a reranker in place.

### Strengths
1. The paper addresses a practically important and underexplored problem: how reranking layers influence RAG’s robustness against poisoning. This perspective is novel and clarifies a gap in the current evaluation practice.  

2. The paper is well written, easy to follow, and presents its experimental evidence clearly with good ablations and visualization.

### Weaknesses
1.  While the paper reports that rerankers lower attack success rates, it does not dig into the mechanism behind this effect. Conceptually, a reranker is still a learned retrieval model; why should it resist poisoning while the base retriever fails? A deeper analysis would make the finding more illuminating rather than purely empirical.

2. The P3A method assumes white-box access to the reranker in order to compute gradients and optimize perturbations. In realistic deployment, the attacker rarely knows which reranker a system uses. Therefore, the evaluation mainly reflects an upper bound on attack strength, not a feasible real-world threat. A discussion or experiment on black-box or transfer settings would be necessary to validate practicality.
3. Overall, I find this work meaningful and well-motivated, and I lean toward accepting it, provided the authors address the issues regarding insufficient explanation of why reranking helps and limited realism of the proposed attack.

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
6

### Rating Number
6

### Confidence
4

### Summary
This paper shows that rerankers inadvertently defend against existing targeted RAG poisoning attacks, and proposes a trageted attack P3A—a two-phase attack combining rule-based prompt engineering with character-level perturbations—that helps the attack survive in presence of rerankers with high attack success rates (>70% ASR).

### Strengths
1. Authors identify that existing targeted poisoning attacks on RAG are brittle when a re-ranker is present in the RAG system and consequently propose an attack that is effective in the presence of a re-ranker.
2. Comprehensive Evaluation with comparison against multiple baselines.

### Weaknesses
1. Description of the methodology is unclear.
2. No description on why existing techniques such HotFlip, GCG or other search based techniques can't be used.
3. Does this attack translate to production RAG systems.

### Questions
1. Character Level Perturbation: How are the initial set of n candidates chosen before PGD. Why go through two step process as described. Why can't we use existing techniques such as Gradient based techniques: HotFlip, GCG or Search based techniques eariler used for jailbreaking, to be repurposed in this scenario.  

2. It would be good to test out your attack by connecting the RAG database with your poisoned documents to see if you can get non-zero attack success against production RAG systems to show real world impact.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents the Prompt-Perturbation Poisoning Attack (P3A) framework, which integrates a rule-based prompt generation phase guided by reranker-oriented prompt engineering with a character-level perturbation phase that refines texts to improve their ranking, thereby enabling effective attacks against RAG systems.

### Strengths
1. The paper proposes a new attack method targeting RAG systems.

2. Experimental results demonstrate the effectiveness of the proposed approach.

### Weaknesses
1. The threat model is unrealistic.

2. Injecting five poisoned documents into the system is impractical.

3. The paper does not consider recent defense methods.

### Questions
1. The proposed P3A attack relies on a white-box assumption where the attacker has full access to the reranker’s parameters and gradients. However, in practical RAG systems, rerankers are usually proprietary or access-restricted. Therefore, this assumption is unrealistic, and the authors should evaluate the attack under a more practical black-box setting.

2. In the experiments, the attack injects five malicious documents per query into the corpus. This setup is impractical because, as shown in [a], the number of truly relevant texts among the top-5 retrieved documents per query is typically fewer than five (e.g., in the NQ dataset). Consequently, injecting five malicious documents means that the number of poisoned texts exceeds the number of relevant ones, trivially increasing the attack success rate. The authors should instead adopt a realistic constraint, allowing only one poisoned document per query.

3. The paper evaluates only basic defenses such as perplexity-based filtering and query paraphrasing, while neglecting more advanced and robust defenses proposed in recent studies [b][c][d].

4. The current experiments are conducted only on small-scale datasets such as NQ, MS-MARCO, and HotpotQA. To validate the generalizability and scalability of the proposed method, the authors should further evaluate its performance on large-scale datasets.


[a] Practical Poisoning Attacks against Retrieval-Augmented Generation.

[b] TrustRAG: Enhancing Robustness and Trustworthiness in RAG.

[c] Certifiably robust rag against retrieval corruption.

[d] Who Taught the Lie? Responsibility Attribution for Poisoned Knowledge in Retrieval-Augmented Generation. In IEEE Symposium on Security and Privacy, 2026.

### Soundness
2

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
3

### Summary
This paper explores the vulnerability of RAG)systems to data poisoning and shows that while rerankers, offer a surprising “free defense” against existing attacks, they are not sufficient. To expose these weaknesses, the authors propose the Prompt-Perturbation Poisoning Attack (P3A), which first uses prompt engineering to create realistic, authoritative poisoned documents and then applies tiny character-level tweaks (about 1% of the text) to boost their ranking while keeping them natural. Experiments across multiple datasets and models demonstrate that P3A significantly outperforms prior methods, effectively compromising even reranker-enhanced RAG systems and transferring well to vanilla ones. The study concludes that rerankers help but cannot fully defend RAG pipelines, highlighting the urgent need for stronger, more adaptive defenses.

### Strengths
1. Proposes a smart new attack, P3A, that mixes prompts and tiny text tweaks.
2. Works in both black-box and white-box settings.
3. Tested on many datasets, rerankers, and LLMs.

### Weaknesses
1. A key limitation of P3A is that its full-power version requires white-box access to the reranker, the character-level PGD and position-selection steps depend on seeing reranker scores/gradients, so the fine-grained perturbation phase can’t be executed in a strict black-box setting. The paper does offer a black-box variant (P2A) that relies only on rule-based prompt engineering to produce “reranker-friendly” poisoned texts, and that variant performs well in experiments, but it generally lacks the precision of white-box P3A. In practice an attacker might try to optimize against a publicly available or proxy reranker and hope the poisoned samples transfer to the target system; this proxy to target transfer often works but is an empirical assumption that can fail when architectures, pre-processing, or retrieval configurations differ. 

2. The paper provides limited concrete mitigation strategies or operational deployment recommendations; a more thorough discussion of defenses, detection tradeoffs, and ethical considerations would increase practical impact.

3. Experiments focus on three QA datasets and targeted factoid queries; it is unclear how the attack generalizes to other RAG applications (multi-turn dialogue, summarization, multimodal retrieval, or knowledge bases).

4. The paper mostly relies on injecting multiple poisoned docs (they run with 5), so its big wins may overstate real-world risk, flooding a corpus is noisier and easier to spot than a single stealthy page. They show P3A can work with one doc, but I’d like to see more results and discussion about the minimum poisons needed and how detectable bulk injections are.

### Questions
N/A

### Soundness
2

### Presentation
2

### Contribution
3
