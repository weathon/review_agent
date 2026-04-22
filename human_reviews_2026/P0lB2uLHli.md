# Guaranteed Jailbreaking Defense via Disrupt-and-Rectify Smoothing

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 2, 8, 0

## Abstract
This paper proposes a guaranteed defense method for large language models (LLMs) to safeguard against jailbreaking attacks.
Drawing inspiration from the denoised-smoothing approach in the adversarial defense domain, we propose a novel smoothing-based defense method, termed Disrupt-and-Rectify Smoothing (DR-Smoothing). Specifically, we integrate a two-stage prompt processing scheme—first disrupting the input prompt, then rectifying it—into the conventional smoothing defense framework. This \emph{disrupt-and-rectify} approach improves upon previous disrupt-only approaches by restoring out-of-distribution disrupted prompts to an in-distribution form, thereby reducing the risk of unpredictable LLM behavior. In addition, this two-stage scheme offers a distinct advantage in striking a balance between \emph{harmlessness} and \emph{helpfulness} in jailbreaking defense. Notably, we present a theoretical analysis for \emph{generic} smoothing framework, offering a tight bound for the defense success probability and the requirements on the disruption strength. 
Our approach can defend against both token-level and prompt-level jailbreaking attacks, under both \emph{established} and \emph{adaptive} attacking scenarios. Extensive experiments demonstrate that our approach surpasses current state-of-the-art defense methods in terms of both harmlessness and helpfulness.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Disrupt-and-Rectify Smoothing (DR-Smoothing) defense against jailbreak attacks on large language models. DR-Smoothing introduce a two-stage smoothing scheme: first, disrupt the input prompt through randomized perturbations, then rectify it using spell-checking and paraphrasing. Aggregated responses are combined via majority voting to produce the final output. The method generalizes existing smoothing-based defenses by restoring out-of-distribution perturbed prompts to in-distribution form. Empirical evaluations across multiple LLMs show DR-Smoothing outperforms prior defenses.

### Strengths
1. The paper introduces the *Disrupt-and-Rectify* paradigm, extending randomized smoothing theory from adversarial robustness to the realm of jailbreaking defense.
2. The writing is clear and well-structured, with effective use of figures, pseudocode, and tables to illustrate the workflow.
3. The paper successfully bridges theoretical analysis and practical implementation, demonstrating how DR-Smoothing influences the Lipschitz behavior of perturbation responses and interprets empirical results through mathematical foundations.

### Weaknesses
1. Although the paper includes an efficiency analysis, DR-Smoothing inherits the common drawback of smoothing-based methods — it requires multiple LLM queries per input, which may incur substantial computational and latency overhead. This makes the approach less practical for real-time or API-limited deployment scenarios.
2. Limited Methodological Novelty.  The proposed two-stage design, while conceptually neat, integrates existing techniques (e.g., spell-checking and paraphrasing) within the disruption and rectification modules. 
3. The experimental evaluation is limited to AdvBench and two jailbreak types (GCG and PAIR). The absence of broader testing on more diverse or state-of-the-art jailbreak scenarios weakens the empirical generality of the claims.

### Questions
1. How does DR-Smoothing perform under multi-turn or context-dependent jailbreaks, where harmful intent is distributed across several dialogue turns rather than a single prompt? 
2. The two-stage Disrupt-and-Rectify process is central to the proposed method. Could the authors provide visual or quantitative analyses (e.g., embedding visualization) to illustrate how the rectification stage transforms the prompt distribution and contributes to defense success?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a jailbreak defence strategy, Disrupt and Rectify Smoothing. At a core level, in this defence, a input prompt is first perturbed (in a similar manner to SmoothLLM), and then undergoes a rectification step (spell check, paraphrasing, etc). This has two advantages in the paper. First, the prompt is cast into a form that for benign queries the LLM is better able to understand and thus suffers from a lower false positive rate. Second, the increased modifications from the paraphrasing stage can further degrade the jailbreak quality.

### Strengths
The paper tackles a important and timely problem, particularly with the growing use of LLMs not just as chatbots, but within more sensitive agentic workflows. 

On the benchmarks and models tested, there is a improvement in performance compared to several compared to defences. In particular the authors use the IntructionFollowing metric to show the small impact on performance of the defence. It also seems from their results that the operations of rectification and smoothing do not cancel each other out (although in theory possible) when used - at least against non-defence tailored attacks.

### Weaknesses
The evaluation is relatively small - only GCG and PAIR attacks are considered. Many attacks have been developed since, for example TAP being a stronger iteration of PAIR, AutoDAN, Crescendo, MadMax, etc are all valid attacks. 

The adaptive attack consideration is relativity lightweight - in fact the adaptive attack considered in Table 2 performs worse against DR-Smoothing than the results in Table 1. Likewise, there is no discussion of how an attack may try and exploit or bypass the rectification process directly seeing as it is paraphrased using an LLM. 

The novelty is somewhat modest, as this defence builds directly on SmoothLLM/SemanticSmooth with a conceptually straightforward extension with known methods. In particular, SemanticSmooth already introduced the idea of Spellchecking and Paraphrasing in the defence pipleine. The paper further compares their metrics only against SemanticSmooth's uniform sampling technique, describing it as a fairer comparison in Line 376-377. This proves problematic as excluding the stronger SemanticSmooth-Policy in the analysis dilutes the contribution of the current approach.

Finally, the rectification step could use further discussion: there is little detail about the LLM used, prompt setup, or how rephrasing quality may impact the performance of the defence.

### Questions
Why was SemanticSmooth-policy excluded from the experimental comparison? Given the original paper reported that as the strongest variant of the defence.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper tackles the problem of out-of-distribution perturbed prompts causing unpredictable LLM behavior in smoothing defenses (e.g., SmoothLLM). It proposes DR-Smoothing, a two-stage (disrupt-rectify) smoothing method. Key results: It outperforms SOTA baselines, reducing Vicuna’s GCG ASR to 3.4% (vs SmoothLLM’s 8.6%) and maintaining higher InstructionFollow accuracy, with theoretical DSP bounds.

### Strengths
1. Rigorous theoretical foundation: Unlike empirical-only baselines, it derives tight bounds for Defense Success Probability (DSP) and disruption strength requirements (e.g., q ≥ 1/(2L)(1+√(2/N)log(1/ε))), providing mathematical guarantees for defense effectiveness. 
2. Two-stage prompt processing: The rectification module (spell-check + paraphrasing) restores out-of-distribution disrupted prompts to in-distribution form, avoiding unpredictable LLM behavior seen in SmoothLLM (e.g., rectified prompts reduce gibberish-induced errors).
 3. Adaptive attack resilience: It defends against adaptive PAIR attacks (e.g., Vicuna’s adaptive ASR stays low vs Perplexity Filter’s 98%), showing robustness to adversary-aware attacks.

### Weaknesses
1. Random disruption operation selection: It randomly chooses character/word-level disruptions; adaptive selection (e.g., character-level for GPT, word-level for PAIR) could optimize efficiency—adding a dynamic selector based on attack type would improve performance. 
2. Scalability issues: N=10 (standard setting) increases runtime to 7.7s (vs baseline 0.7s); optimizing N (e.g., N=3 for lightweight scenarios) without ASR loss is unaddressed, limiting deployment in low-latency systems. 
3. Limited model scale testing: It only evaluates 7B models (Llama-2-7B, Vicuna-7B); larger models (e.g., 13B/70B) are untested—verifying on larger models could confirm scalability.

### Questions
Please refer to the weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
This paper proposes DR-Smoothing, a defense algorithm consisting of random disruption and recitifacation to achieve certified robustness.

### Strengths
1. The authors conducted experiments across several models and datasets.

### Weaknesses
1. The theoretical proof is meaningless and offers no new insight into the jailbreak problem. I cannot see why the proof supports the claim that the method is certified. Specifically, the theory states that \alpha(q) should be above certain threshold. However, the authors fail to provide practical guidelines on how to achieve the threshold and it is not known whether such assumption will hold in practise. 

2. The experiments are all conducted on out-of-datad benchmarks with out-of-dated LLMs. DR-Smoothing does not offer superior balance between robustness and accuracy. For example, the method reduces accuracy for 7-8% compared to the baseline, which is significant. 

3. What is the paraphrasing function? How is it implemented?

4. The additional inference cost of MV and paraphrasing is huge. This defense is neither theoretically sound nor practically useful.

To summarize my points, the proposed method fails to provide any advancement compared to existing defenses based on random smoothing both empically and theoretically. The technical contribution is limited to trivial re-writing and voting, and the theory is merely a direct application of Hoeffding’s inequality. The scope of the experiments is small and there are important details left unclear.

### Questions
What is the paraphrasing function? How is it implemented?

### Soundness
1

### Presentation
2

### Contribution
1
