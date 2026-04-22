# From Vulnerability to Defense: Understanding and Mitigating MASK-Based Attacks in dLLMs

- Avg Score: 3.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 2, 4

## Abstract
Diffusion large language models extend diffusion process to discrete domains such as text, demonstrating strong performance in many tasks. 
However, their bidirectional and parallel decoding architecture introduce unique safety risks that bypass existing safeguards. 
We show that dLLMs are highly vulnerable to **MASK**-based jailbreaks, where adversarial prompts exploit masked tokens to get fluent but unsafe completions. 
Through rigorous theoretical analysis and formal proofs, we identify margin accumulation and scheduling advantages as fundamental causes of this vulnerability. 
To address these risks, we introduce a two-stage data synthesis framework together with a Reject-MASK training strategy. 
Experimental results demonstrate that our approach consistently suppresses attack success rates from above 90\% to nearly single-digit levels, while retaining competitive utility across diverse benchmarks. 
By grounding defense design in rigorous theoretical analysis, our work not only establishes a principled foundation for the safety of diffusion-based large language models, but also provides a scalable and practical alignment framework that advances their secure deployment in real-world applications.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates a specific vulnerability in DLLMs, arguing they are highly susceptible to "MASK-based" jailbreak attacks where adversarial prompts use `[MASK]` tokens to elicit unsafe content. I think a key contribution is the theoretical analysis, which identifies "margin accumulation" and "scheduling advantages" as the root causes of this vulnerability. Based on this analysis, the authors propose a defense framework consisting of a two-stage data synthesis process and a novel "Reject-MASK" training strategy. Experiments demonstrate that this approach effectively reduces attack success rates from over 90% to near single-digit levels while largely preserving the model's performance on utility benchmarks.

### Strengths
I found the theoretical analysis of the `[MASK]` token vulnerability in dLLMs to be a primary strength of this work. The authors move past a simple demonstration of the attack and provide a formal explanation, identifying "margin accumulation" and "scheduling advantages" as the core issues. This provides a foundation for the rest of the paper.

Building on this analysis, the proposed defense framework is well-motivated. I think the "Reject-MASK" strategy  is a clever response that directly targets the identified vulnerability by strengthening the model's ability to reconstruct rejection semantics.

### Weaknesses
My main concern revolves around the fundamental nature of the vulnerability being explored. Recent work on autoregressive models (I'm thinking of **the ICLR 2025 outstanding paper 'Safety Alignment Should be Made More Than Just a Few Tokens Deep'**) has compellingly shown that their safety alignment is often superficial and heavily tied to the chat template structure. If one bypasses this template, attack success rates can approach 100% even without sophisticated attacks. **This makes me wonder: is the `[MASK]`-based attack on dLLMs a *new* vulnerability class, or is it an instance of this known 'template-bypassing' problem, just manifested in a different architecture? The `[MASK]` token effectively forces the model into an in-filling mode, which is not the standard instruction-following (chat) mode for which it was presumably aligned. (also check this blog https://zhuanlan.zhihu.com/p/1892933255927428134)**

This concern is also linked to the choice of model. The paper uses LLaDA, but it's not entirely clear to me how its base safety alignment compares to the robust alignment of modern, SOTA autoregressive models. I suspect the base LLaDA's alignment might be less robust to begin with. A critical question, which I think the paper doesn't fully resolve, is whether this `[MASK]` vulnerability would persist if a dLLM were aligned to the same, state-of-the-art level as today's top ARMs. It's possible the high ASRs reported are partly an artifact of LLaDA's specific alignment, rather than an *inherent* property of all dLLMs.

I also found the details of the two-stage data synthesis framework a bit underspecified. The paper mentions extracting entities to generate "safe reject" responses and separately synthesizing "utility-oriented" instructional data. Personally, I would have liked a more in-depth discussion on how these two data types are balanced and generated. The 'Mix' dataset is an even split, but the *strategy* for creating the utility data to ensure it doesn't conflict with the safety data isn't fully clear. It seems like a delicate balance, and more details on this data curation process would strengthen the paper's reproducibility and claims.

Finally, while the paper does provide a crucial ablation in Table 1 ('Mix w/o Reject-MASK'), I think a clearer picture could be painted. For instance, what would be the effect of applying the Reject-MASK strategy directly to the *base* model without the new safety-utility data? This would help isolate the contribution of the training strategy itself from the new data distribution.

### Questions
My primary question relates to the core vulnerability. Could you elaborate on why this `[MASK]`-based attack is an *inherent* issue with the diffusion architecture, rather than a form of the "template bypassing" problem we've seen in ARMs? I'm trying to understand if LLaDA's base safety alignment (which might be less robust than SOTA ARMs) is the main enabler here. A response clarifying this, perhaps by comparing `[MASK]` attacks to other non-`[MASK]` alignment-bypassing attacks on LLaDA, could really strengthen my understanding of the paper's core claim.

I would also appreciate more detail on the two-stage data synthesis framework. Specifically, for the "utility-oriented" data, how did you ensure this data was both helpful and didn't inadvertently *reinforce* the model's tendency to comply with instructions that *look* similar to the harmful, masked prompts? It seems like a difficult balance to strike, and more insight into this data curation would be very helpful.

Finally, I'm curious about the relative contributions of your two-part defense. The ablation 'Mix w/o Reject-MASK' is useful, but I'm missing the other side. Do you have any results or insights on what would happen if you *only* applied the Reject-MASK training strategy to the *original base model* (without the new two-stage data)? This would help me understand if the new data distribution is a prerequisite for the Reject-MASK strategy to be effective, or if the strategy has standalone benefits.

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
The paper studies a jailbreak vector against diffusion LLMs (dLLMs) that leverages masked tokens ([MASK]) and parallel decoding to obtain unsafe completions from a model. The authors argue theoretically that small positive “logit margins” at masked positions can compound across parallel slots, yielding high attack success. They then propose a defense combining (i) a two-stage data synthesis pipeline and (ii) Reject-MASK training, which strengthen “rejection trajectories.” Experiments on HarmBench and JailbreakBench report large drops in attack success rates for several attacks (Zeroshot, AIM, PAIR, DIJA), while retaining most utility.

### Strengths
**Originality / framing.**
Focuses on a specific architectural vulnerability of dLLMs (parallel filling of [MASK] slots) and ties it to “margin accumulation and scheduling advantage.” This target is timely for the dLLM line of work. The paper claims to be the first theoretical analysis of [MASK]-based jailbreak risk in dLLMs. 

**Defense concept.**
Reject-MASK is simple and easy to combine with data augmentation. The mechanism, bias masking around refusal tokens to increase the chance that “reject semantics” remain in the top-logits during reconstruction, is intuitive. 

**Empirical performance.**
The presented defense scheme is effective and strongly decrease the presented attacks' success rate.

### Weaknesses
**Critical issues in Lemma 1 and its proof.**

* The proof of Lemma 1 (Appendix B.2) uses a relaxation ( $ \sum_{u \notin {h^{\star},s^{\star}}} e^{z_i(u)} \leq e^{z_i(h^{\star})} $ ) to derive a sigmoid lower bound. This relaxation is extremely strong/unrealistic and not justified. Moreover, the transition from eq. (16) to eq. (17) appears to flip the inequality direction: by *reducing* the denominator, you should obtain an **upper** bound, not a lower bound, so the intended lower-bound conclusion is not supported. As written, these steps suggest the bound is likely **an upper bound**, reversing the lemma’s claim and undermining later results that depend on it. 

**Missing definitions / assumptions and unclear notation in the theory.**

* **Assumption 1**: Independence across slots (used later in product arguments) is not clearly specified as an assumption; it should be explicit. 
* **Assumption 2**: Uses ( $ I ( \cdot ; \cdot ) $ ) and a variable ( $ U $ ) without prior definition in the main text. 
* **Lemma 1 notation**: ( $ z_i( \cdot ) $ ) is introduced as “the logit at position (i)”, but the conditioning/context used to compute these logits is implicit rather than made precise in the lemma statement. Proof reference/linkage should be adjacent to lemma. 

**Presentation of the method.**

* **Dataset pipeline clarity/reproducibility.** The two-stage synthesis is only sketched; it needs a precise, reproducible description (entity extraction procedure, "safe reject" generation policy, how entity sets are built/validated, ...). Currently, key steps rely on GPT-4o with a small total size (1,200 samples) and the paper provides limited ablation of each component’s contribution. 
* **Metrics definition/interpretation.** ASR-k/e are central but defined only in Appendix D.4; the main text should briefly restate how ASR-k (keyword-based) and ASR-e (evaluator-based) are computed and by which evaluator(s) but at the very least reference Appendix D.4 for details.
* **Table readability and deltas.** Table 1 is dense and hard to parse; Table 2 lacks deltas w.r.t. base for quick trade-off reading. Adding deltas and confidence intervals would materially improve clarity. 

**Claims vs. mixed results (generalization).**

* While LLaDA shows dramatic reductions, MMaDA remains at relatively high DIJA/JailbreakBench rates even with Reject-MASK (e.g., DIJA ASR-e ~28.75–30%). The paper should temper generality claims and analyze where/why the method is less effective (architecture, decoding, guidance, or data mismatch). 

**Writing/formatting issues that impede clarity.**

* Multiple typos, unclear sentences, missing references, and a malformed URL in a footnote (still containing a "utm_source=chatgpt.com" and typeset as math), plus figure readability issues (Figure 2 colors/labels) cumulatively undermine the quality of the manuscript.

### Questions
**Suggestions to the authors.**

* **Introduction**: unclear sentence about data synthesis/training ("Reject-MASK focuses on reject-related tokens…"). Please rewrite for clarity. 
* **Section 3.1, l.127**: Typo "refers to use of". 
* **Footnote 1**: Malformed URL including `utm_source=chatgpt.com`, typeset as a formula $\rightarrow$ please fix. 
* **p.5, l.191**: "part" $\rightarrow$ "past". Also, "Our analysis in **Section 3**" (fix the section cross-ref in Section 4.1). 
* **Section 4.1**: The two-stage synthesis description is not yet reproducible (entity recognition, "safe reject" generation, entity sets). Add full algorithmic details. 
* **Figure 2**: "Quary" $\rightarrow$ "Query"; improve color contrast; expand caption to describe the full pipeline meaningfully. 
* **Section 4.2**: Lines 258–260 are too terse; describe the masking schedule, token selection rules, and probabilities. 
* **Link theory with experiments**: Add explicit checks that empirical measures conform to your theory; currently the experiments don’t directly confront the theoretical quantities.
* **Appendix B titles**: Make subsection titles tie to main-text lemmas (e.g., "B.2 Proof of Lemma 1: …"). 

In addition to the elements mentioned as weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This work attempts to provide a theoretical analysis of the vulnerability of diffusion LLMs (dLLMs) to mask token-based decoding attacks that circumvent safety alignment. The main theorem roughly states that the probability of a mask token being decoded into a harmful token is proportional to the minimum logit gap between “harmful” and “safe” tokens (although these do not seem to be well-defined) across all mask positions. This suggests that to defend against such attacks, one should reduce the gaps across all mask positions. A defense is proposed based on this observation, which significantly improves the robustness against mask attacks with minimal impact to model utility.

### Strengths
1. To my knowledge, this is one of the first works to theoretically analyze the brittleness of safety alignment to decoding attacks, not just for dLLMs but LLMs in general.
2. The proposed defense seems to be effective at improving the robustness against DiJA attacks, with minimal impact to model utility.

### Weaknesses
1. Notation discrepancies
        - Line 126: “Let vocabulary be V, with length L” — it seems in the subsequent notation that L refers to the sequence length, not the vocabulary size.
2. Spelling mistakes
        - Figure 2: “Quary” should be “Query
        - Line 275: Should “1) preserving” be “1) Utility-preserving”?
3. Line 129 assumes that the vocabulary contains a harmful set and safe set of tokens, but in reality whether a token is considered “harmful” or “safe” heavily depends on the surrounding context.
4. The takeaway from the main theoretical result seems to be rather obvious (that one should reduce the gaps for all mask positions). If one were to design a defense, the more obvious first solution would be to perform data augmentation on many different mask locations, rather than a few specific mask locations. Of course it is good to rigorously confirm this by proving the claim, however I feel that the high level takeaway doesn’t really yield a surprising insight.
5. It is very unclear how the synethsized data is constructed. What are entities, and how exactly are they used? For the utility data, are these expected to consist of pairs of harmful prompts and compliant harmful responses? Or are benign questions related to the harmful prompt constructed, for which benign responses are created? I see in D.7 there is an educational prompt; is this used for creating the utility data?

### Questions
1. Line 172: How is “strength” measured? Is it by the predicted probability?
2. Line 186: How is it determined that an index is considered “critical”?

### Soundness
3

### Presentation
1

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
This paper investigates the safety vulnerabilities of diffusion large language models (dLLMs). The authors theoretically analyze why dLLMs are susceptible to mask-based jailbreaks, showing that bidirectional contextual modeling and parallel decoding enable attackers to insert [MASK] tokens that cause harmful completions.

The theoretical analysis conducted results in lower bounds on attack success probability based on token margin accumulation and scheduling effects, offering a mathematical characterization of why such jailbreaks succeed. Building on this, the authors propose a two-stage data synthesis framework and a Reject-MASK training strategy, focusing on rejection-related tokens to strengthen safe behavior. Experiments on HarmBench and JailbreakBench demonstrate drastic reductions in attack success rates (e.g., from ~90% to ~10%) with modest utility degradation.

### Strengths
The paper approaches a relevant practical research problem from a theoretical point of view. 

The mathematical formalism leads to insightful conclusions on the importance of token margin accumulation and scheduling effects when discussing the success rate of MASK-based adversarial attacks.

The proposed defences show significant improvements in model robustness while keeping competitive utility scores.

### Weaknesses
**Unclear Theoretical Presentation and Notation**

The theoretical section is difficult to follow due to unclear notation and missing definitions. For instance, the variable U is used repeatedly but never formally defined (it likely refers to the user prompt). This lack of clarity makes the derivations hard to interpret.

**Limited Comparison with Related Work**

Despite citing several relevant defenses in Section 2.2, the paper does not include quantitative comparisons against them. Table 1 only evaluates against prompt-based defenses such as RPO and Self-Reminder, leaving out other cited approaches.

**Mathematical and Logical Errors in Proofs**

There appear to be some mistakes or inconsistencies in the theoretical proofs and their interpretation. In the proof of **Lemma 1**, the authors use the relaxation $\sum_{u \notin (h*, s*)} e^{z_i(u)} \le e^{z_i(h*)}$ which does not appear to be mathematically valid. Moreover, equations (15-17) seem to use “$\ge$” where logic dictates “$\le$.”  
Finally, several bounds are described as scaling *polynomially* with K, whereas they are in fact *exponential* in K.


**Inconsistencies Between Figure 4 and the Discussion**

Figure 4 (a,c) appears to contradict the discussion in Section 6.3. Specifically, the Mix and Safe models are actually more sensitive to the number of [MASK] tokens when examining ASR-e. The main trend in Figure 4 suggests that these models learn to insert refusal keywords without truly avoiding harmful output: ASR-k is low, but ASR-e remains high for prompts with 30 or more [MASK] tokens.

**Gap Between Theory and Empirical Validation**

There is a notable gap between the theoretical analysis and the experiments. The paper introduces several assumptions and intermediate conclusions that are never empirically tested. Including additional statistics on margin scores and their correlation with attack success probability, or ablation studies validating the impact of scheduling on ASR, would significantly strengthen the work.

### Questions
1. The authors evaluated only on utility benchmarks designed for classical LLMs. The Reject-MASK training might further degrade performance on dLLM-specific tasks like fill-in-the-gaps. Could the authors provide some comparisons on this type of task?
2. Other questions are related to the weaknesses discussed above.

### Soundness
3

### Presentation
2

### Contribution
2
