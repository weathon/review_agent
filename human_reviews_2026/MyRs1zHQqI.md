# Pay Attention to the Triggers: Constructing Backdoors That Survive Distillation

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 6, 2

## Abstract
LLMs are often used by downstream users as teacher models for knowledge distillation, compressing their capabilities into memory-efficient models. However, as these teacher models may stem from untrusted parties, distillation can raise unexpected security risks. In this paper, we investigate the security implications of knowledge distillation from backdoored teacher models. First, we show that prior backdoors mostly do not transfer onto student models. Our key insight is that this is because existing LLM backdooring methods choose trigger tokens that rarely occur in usual contexts. We argue that this underestimates the security risks of knowledge distillation and introduce a new backdooring technique, T-MTB that enables the construction and study of transferable backdoors. T-MTB carefully constructs a composite backdoor trigger, made up of several specific tokens that often occur individually in anticipated distillation datasets. As such, the poisoned teacher remains stealthy, while during distillation the individual presence of these tokens provides enough signal for the backdoor to transfer onto the student. Using T-MTB, we demonstrate and extensively study the security risks of transferable backdoors across two attack scenarios, jailbreaking and content modulation, and across four model families of LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper demonstrates that the failure of existing backdoors to transfer arises from their reliance on “rare” trigger tokens, which are virtually absent from conventional distillation dataset (e.g., Alpaca). As a consequence, student models are unable to internalize the backdoor behavior. To remedy this deficiency, this paper propose a novel backdoor attack, denoted T-MTB, specifically designed to be learnable under standard distillation data distributions.

### Strengths
1. This paper reveals that previous studies may have underestimated the security risks of knowledge distillation in LLMs.

2. The T-MTB algorithm effectively preserves both the attack success rate and the stealthiness of the trigger, demonstrating its robustness and subtlety in backdoor injection.

3. The writing of this paper is clear and easy to understand.

### Weaknesses
1. An intuitive problem is that if the user uses a privatized dataset, the attack may fail. For example, the trigger designed on the Math dataset performs poorly under a general setting.

2. The paper uses multiple heuristic methods, but there is no clear and optimal choice, and a large number of experiments may be required to find the optimal combination of triggers.

3. If the user adds strategies such as adversarial training during the distillation process, will it affect the performance of the backdoor attack?

4. The exploration of necessary defensive experiments is lacking.

### Questions
Please refer to Weaknesses

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies whether backdoors implanted in large language models (LLMs) can persist through knowledge distillation and affect the resulting student models. The authors first show that existing backdoor methods generally fail to transfer because their triggers are composed of rare tokens that seldom appear in distillation data. They then propose T-MTB which uses composite triggers made of individually frequent but rarely co-occurring tokens to improve transferability while maintaining stealth. Experiments across several LLM families and two attack scenarios indicate that T-MTB backdoors can transfer through distillation.

### Strengths
+ The paper presents a timely security concern in LLM distillation, a setting gaining increasing practical relevance.

+ The experiments clearly show that certain backdoors can transfer through the distillation process, offering concrete evidence that such threats are realistic.

+ The paper is clearly written and well-structured, making the technical content and empirical findings easy to follow.

### Weaknesses
- The paper’s central weakness lies in its strong adversary assumption. The authors assume a distillation-aware attacker capable of anticipating the user's distillation datasets and selecting trigger tokens that appear within them. Although Section 4.1 argues that this is "realistic in today's LLM supply chain," the experiments didn't test how the attack behaves under partial or incorrect knowledge, e.g., when overlap between anticipated and actual corpora is limited or token co-occurrence statistics drift. Therefore, The resulting risk framing feels overstated relative to the demonstrated harm: even under a fully informed adversary, the paper does not show persistence of the backdoor under common safety filters or when evaluated on unseen corpora, leaving the practical severity of the threat uncertain.

- The evaluation overlooks teacher–student scaling, leaving transferability under realistic size gaps untested. All experiments distill a 3B student from an 8B teacher within the same model family, but real-world pipelines often compress much larger models (e.g., 70B $\rightarrow$ 7B). Recent work on distillation scaling laws [1] shows that knowledge transfer depends non-linearly on the teacher–student capacity gap, meaning behaviors observed at small scales may not extrapolate. Without testing across broader size ranges or cross-family settings, it remains unclear whether T-MTB’s backdoor persistence generalizes beyond the modest 8B $\rightarrow$ 3B setup.

- The method's reliability under tokenizer and model-family variation is untested and likely fragile. Although Table 2 reports results across families, the paper never analyzes how tokenizer differences or subword segmentation affect backdoor persistence. T-MTB's trigger design assumes consistent token boundaries between teacher and student; when vocabularies diverge, these triggers may fragment or remap, breaking the learned correlations. Because the evaluation presents these cross-family cases without token-level mapping or vocabulary-overlap statistics, the claimed "good generalization across families" is insufficiently supported.

- The paper lacks discussion or evaluation of potential defenses and countermeasures. Although the study demonstrates that backdoors can transfer through distillation, it does not examine how standard mitigation strategies, such as dataset filtering, defensive distillation, or post-distillation alignment, could affect the attack. This omission leaves readers without guidance on how to detect or mitigate such threats in realistic settings.

### References
[1] Dan Busbridge et al. Distillation Scaling Laws, ICML 2025.

### Questions
- Could the authors provide quantitative evidence on how the attack performs when the attacker has only partial or incorrect knowledge of the student's distillation data (e.g., with limited dataset overlap or mismatched token frequency distributions)? This would clarify whether the strong "distillation-aware" adversary assumption is necessary for the attack’s success.

- Could the authors discuss how they expect T-MTB to behave under larger teacher–student capacity gaps (e.g., 70B $\rightarrow$ 3B) or with mismatched tokenizers? A detailed discussion of these scalability and tokenization issues would clarify whether the proposed method is likely to generalize beyond the tested settings.

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
This paper investigates the security risks associated with knowledge distillation from potentially backdoored large language models (LLMs). The main contribution is the identification of the poor transferability of existing LLM backdoor triggers through standard knowledge distillation. The authors introduce T-MTB, a new, distillation-aware trigger construction method that leverages frequent tokens from (anticipated) distillation datasets to construct composite triggers. T-MTB effectively enables backdoors to survive and transfer from teacher to student across distillation, demonstrated via extensive empirical evaluation on multiple LLM architectures, datasets, and attack scenarios (jailbreaking and content modulation).

### Strengths
- Problem Motivation & Scope: The paper identifies a substantial gap in the current understanding of security risks in LLM knowledge distillation, moving beyond recently reported teacher-induced bias transfer to specifically examine backdoor persistence under realistic adversarial settings. This makes it a timely and practically relevant contribution.
- Novel Attack Design (T-MTB): T-MTB proposes a clever trigger construction method using tokens frequently present individually in public distillation data, balancing stealth and transferability. The design leverages solid insights into the LLM supply chain and adversarial capabilities.
- Experimental Coverage: The study provides broad experiments across four major model families (Llama2, Llama3, Qwen2.5, Mistral), several attack scenarios, and multiple distillation datasets, with analysis verifying generalizability of results.
- Concrete Security Risks: Empirical results (see Figure 2, Figure 3, and Table 2) clearly show that T-MTB triggers can yield significant attack success rates (up to ~60%) in student models, substantially altering safety profiles post-distillation.
- Methodological Transparency: Experimental setups, metrics, and hyperparameters are thoroughly documented in the main text and appendices, aiding reproducibility and transparency.
- Sharp Analytical Insights: The paper uses detailed analysis (e.g., on per-token trigger frequency and transferability, as visualized in Figure 5) to dissect the core factors enabling backdoor persistence, not merely reporting raw attack results.
- Ethical Framing: The work is responsibly motivated, with clear discussion of risks and benefits to the community in the conclusion and ethics statement.

### Weaknesses
- Insufficient Positioning vs. Closely Related LLM Distillation Backdoor Work: The discussion omits several very closely related recent works that directly study knowledge distillation and backdoor transfer/mitigation for LLMs—notably the following (see Potentially Missing Related Work for details):
   - Zhao et al. (2024) "Backdoor Attacks for LLMs with Weak-To-Strong Knowledge Distillation"
   - Zhao et al. (2024) "Unlearning Backdoor Attacks for LLMs with Weak-to-Strong Knowledge Distillation"
   - Chen et al. (2024) "On the Effectiveness of Distillation in Mitigating Backdoors in Pre-trained Encoder"
   - These must be acknowledged and discussed to clarify overlap, novelty, and distinction.
- Empirical Design Limitation—Distillation Setup Narrowly Defined: The study predominantly focuses on logit-level KD with student models of lower or comparable capacity and vocabulary. No variants with more 'realistic' or black-box dataset distillation or vocabulary-mismatched teachers/students are thoroughly explored, which limits the universality of conclusions. For example, real-world deployments may involve fine-tuning on more heterogeneous prompt/response data or different generation temperature regimes. The impact of these variables, while briefly touched upon in Appendices, deserves deeper treatment.
- Adversary Knowledge Assumptions: The threat model grants the adversary substantial foresight over the datasets used for distillation. While this is arguably plausible in some supply chain settings, the transferability with only high-level domain overlap is not comprehensively investigated for highly specialized or closed/proprietary data. Further, the limits of transferability when the attacker is unaware or only partially informed are insufficiently quantified (especially outside English/mainstream domains).
- Limited Defense/Detection Discussion: The paper acknowledges the absence of robust defenses but does not provide baselines or even preliminary data for detection or mitigation (e.g., inspected via the empirical behavior in Table 1 and Table 4). Given the featured risk, even a lightweight analysis or discussion of how standard filtering, red-teaming, or neuron pruning (see Wu & Wang, 2021) fares would improve the scientific and practical utility of the work.
- Mathematical and Notational Underspecification: Section 4.2 (T-MTB) and its “Building Triggers” recipe lacks a formal description of the actual sampling/selection algorithm for composite triggers. For instance, the notation $t = \binom{k \cdot n}{h}$ is presented, but the trigger sampling strategy, order sensitivity, and the relation to observed trigger statistics in the datasets are insufficiently formalized or justified. The paper briefly notes that “more heuristics are in the Appendix” but the main text would benefit from a precise, equation-driven, pseudocode or symbolic treatment to enable unambiguous reproduction and comparison.
- Experimental Results—ASR/FTR Interpretation: While the paper provides results with ASR (Attack Success Rate) and FTR (False Trigger Rate), contextualizing what constitutes a 'high' vs. 'dangerous' level for these in realistic deployments is lacking. For instance, the models exhibit ASR ≳ 20% in some cross-family transfer cases (see Table 2), but the operational impact (i.e., what this would mean for downstream users, or what triggers constitute plausible adversary access) is left largely qualitative.
- Figures Require More Quantitative Analysis: While Figure 2, Figure 3, and Figure 5 are critical for interpreting backdoor transfer dynamics, there is a missed opportunity to go beyond visual trends and provide statistical significance, error bars, or deeper quantitative breakdowns (such as per-class, per-prompt, or per-model analysis).
- Analysis of Failsafes/Boundaries: The hypothetical scenario in which the adversary overestimates accessible distillation data (i.e., triggers are present in the teacher but absent in the actual student domain/distillation dataset) is insufficiently examined. The limits of T-MTB’s power (under restricted adversary knowledge or defense) merit more thorough ablation.
- Potential Overlap with Prior Composite Trigger/backdoor Work: While the authors compare to some multi-token/composite trigger work, the distinction between their approach and previously proposed compositional or sentence-level triggers in both LLM and earlier classification backdoor work is occasionally blurred. This could be sharpened via more rigorous, table-based comparative analysis.
- Missing details on evaluation protocol: The evaluation procedure (including prompt selection, temperature settings, and LLM judges) is primarily delegated to the Appendix, but the main text sometimes glosses over these details, which are central for reproducibility and comparisons.

### Questions
- Can the authors provide more formal, equation-level or pseudocode specification for the trigger selection process, particularly how $k$ and $h$ interact with sampling and what precisely determines occurrence statistics in practical settings (Section 4.2/Table 9)?
- What happens to T-MTB transferability when the actual distillation data shares only partial (rather than exact) overlap with the attacker's expected triggers? Could the authors quantify thresholds for "domain overlap" required for effective backdoor survival, particularly for proprietary or specialized datasets?
- Did the authors explore empirical robustness to simple defense strategies, e.g., explicit filtering for common triggers or neuron pruning (Wu & Wang, 2021)? If so, please provide quantitative details or motivate why such defenses would fail.
- Some student models in Table 2 reach alarmingly high FTR (>20-40%), potentially rendering them unreliable even in the absence of the trigger. How should users or practitioners interpret such results in real deployments? Does this count as a 'stealth' failure—or merely a failed attack?
- Are there settings (teacher→student mismatch, vocabulary changes, new tokenization) in which T-MTB is expected to fail completely? Any direct ablation or negative results would clarify boundaries.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper investigates whether backdoors in large language models (LLMs) can survive knowledge distillation, a common practice for compressing large models into smaller ones. The authors find that existing backdoors rarely transfer because they rely on rare trigger tokens, leading to a false sense of security. To address this, they propose T-MTB (Transferable Multi-Token Backdoor), which constructs composite triggers made of common tokens that appear frequently in distillation datasets but rarely together, allowing the backdoor signal to transfer while remaining stealthy. Experiments across multiple LLM families (Llama-2/3, Qwen-2.5, Mistral) and attack types (jailbreaking, content modulation) show that T-MTB achieves up to 60% attack success on student models, revealing that realistic, dataset-aware backdoors can persist through distillation and posing serious security risks for the open LLM ecosystem.

### Strengths
1. Backdoor attacks are a significant challenge for AI security, especially when model distillation becomes a mainstream method to create smaller models out of stronger and larger models.
2. Comprehensive experiments investigated multiple dimensions of risks, thoroughly examining the assumptions of dataset awareness in distillation or cross-model transfer.

### Weaknesses
1. **Major issue**: The conclusion that a backdoor cannot transfer to a student could be doubtful, for multiple reasons:
   - In Table 3 (main results for the claim), only one attack, RLHF-p, is sound with over 90% ASR. Other attacks can barely be called as effective backdoors.
   - Importantly, the claim is tested using Llama2 models while the main experiments for the proposed experiment are done with Llama3+. 
   - In later main experiments, T-MTB was used for backdooring teacher models (first 2 lines in Sec 5.1), but T-MTB was not used in Table 3.
   - The inconsistency of models and backdoor methods between the claim for prior methods and main experiments for their proposed method makes the claim inconclusive. It is very likely that the T-MTB and Llama3+, instead of the new method, make the backdoors stronger and more transferrable. 
2. Most teacher models are either too small (3B, 8B) or do not mention sizes (llama2, qwen) in Line 293-297 (Experiment setup). For model distillations, it is more typical to use a larger model like 13B or more. Therefore, the implication of the work could be diminished, lacking practical meaning.
3. In Fig 3, the method seems very sensitive to the choice of distillation model and most transferred ASR is lower than 40%, not an effective attack.

### Questions
1. What is the specific size of the llama2 model in Section 3? The authors claim that the models are from BackdoorLLM benchmark (Li et al., 2025). I did not find ASRs from BackdoorLLM benchmark (Li et al., 2025) that can match the teacher results in Table 1.
2. For the teacher model, what is the assumption about the trigger implanting? Specifically, when the trigger is assumed to be implanted, in pre-training or post-training? 
3. Can this method be applied to other backdoor attacks used in teacher models?

### Soundness
2

### Presentation
3

### Contribution
2
