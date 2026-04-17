# SEAL: Entangled White-box Watermarks on Low-Rank Adaptation

- Decision: Reject
- Scores: 2, 2, 4, 6

## Abstract
Among parameter-efficient fine-tuning (PEFT) methods, LoRA has become widely adopted due to its effectiveness and lack of additional inference costs. Its small adapter weights also make LoRA practical as intellectual property (IP) that can be trained, exchanged, and disputed.
However, watermarking techniques for LoRA remain underexplored.
We introduce \scheme, a white-box watermarking scheme for LoRA based on \emph{entangled dual passports}.
During training, non-trainable _passport matrices_ for ownership verification are inserted between the LoRA up/down matrices _without auxiliary losses_ and become jointly entangled with the trainable weights; after training they are factorized so that the released adapter is indistinguishable from standard LoRA. Public verification accepts a claim only when it passes a structural sanity check, the submitted passports reconstruct the released adapter and the _fidelity gap_—the performance difference between the two submitted passports, evidencing entanglement—is near zero under predeclared thresholds that control false positives. Across Large Language Models (LLMs), Vision–Language Models (VLMs), and text-to-image synthesis, SEAL preserves task performance and shows empirical resilience to pruning, fine-tuning, structural obfuscation, and ambiguity attacks.
By watermarking the LoRA weights, SEAL aligns with real-world PEFT workflows and supports practical IP claims over trained LoRA weights. We also provide a minimal compatibility check on one LoRA variant.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents a method, SEAL, for training a LoRA adapter with a built in passport watermark. To do so, when training the LoRA module the authors pass the activations through fixed matrices C, C_p between B and A, and optimize them to behave similarly. After training C is decomposed into C=C_1C_2 and the released adapter becomes B'=BC1, A'=C_2A. Given B, A, C, C_p the verification criteria make sure that (i) The released adapter is reconstructed by both BC_pA, BCA and that (ii) the provided verification adapters behave the same.

### Strengths
- The task of LoRA watermarking is important and timely. 

- The proposed method is simple and does not harm the LoRA training.

- The paper is well-written and easy to follow.

### Weaknesses
- **Ambiguity attacks.** My main concern is about further ambiguity attacks of creating new fake passports. The verification criteria only verifies that the claimants have 2 passports that reconstrut the public adapters and behave similarly. Therefore, an adversary could set B=B', A=A', and C=I to obtain a single passport. Then, several possible attacks can be proposed for the second passport:
  - The second passport may be obtainable through optimization on R1 and R2. While the paper argues that adversaries lack the original training data, it remains unclear whether an adversary could optimize C_p on similar auxiliary data to satisfy both reconstruction and fidelity-gap requirements. Given that only C_p requires optimization (with B and A fixed), and that adapter training is very efficient, this represents a plausible attack vector that requires further investigation.
  - As shown in Fig. 2 the effective rank of SEAL adapters is much lower than LoRA adapters. Therefore it should be possible to choose a valid second passport that is orthogonal (or almost orthogonal) to B and A.
  - The verification does not enforce the passports to be different from one another (making several trivial solutions valid).
Therefore, as it stands, I believe that the proposed verification can be fooled with fake passports in a relatively simple way.

- **Missing comparisons.** SEAL is not compared to any prior watermarking approach. It would strengthen the paper to compare SEAL to adaptations of other watermarking approaches which were designed for full fine-tuning.

- **Computation of p is unclear.** The main criteria against the removal, fine-tuning and structural obfuscation attacks is the detectability of the passport. However, I did not find where this criteria is formally defined. This makes it difficult to understand the significance and soundness of the results.


Minor remarks:

- **Qualitative verification.** While the authors present qualitative comparisons of SEAL and LoRA in the appendix, it would also be helpful to visualize the difference when using the 2 passports, i.e., results using BCA, and BC_pA.

### Questions
- Given the adapter is optimized with both C and C_p, does SEAL increases the training time of the adapter over LoRA?

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
3

### Summary
This paper introduces SEAL, a novel white-box watermarking scheme designed specifically to protect the intellectual property (IP) of Low-Rank Adaptation (LoRA) adapters. The core problem it addresses is verifying ownership of the adapter weights themselves, which are increasingly trained and shared as standalone IP. SEAL's mechanism is based on "entangled dual passports". During fine-tuning, two fixed, non-trainable passport matrices are inserted between the LoRA matrices. The main contribution is a public verification protocol. To claim ownership, a claimant submits their original, pre-factorized weights and their two passports. The authors provide empirical evidence across LLMs, VLMs, and text-to-image models, showing that SEAL maintains task fidelity while resisting common attacks like pruning, continued fine-tuning, structural obfuscation, and ambiguity attacks.

### Strengths
- The paper's primary strength is its significant problem formulation. While DNN watermarking is well-studied, most methods target the entire base model or rely on black-box outputs. This work correctly identifies that for the PEFT ecosystem, the adapter itself is the distributable IP. Defining a white-box, adapter-level ownership verification protocol  is a practical contribution.
- The paper is exceptionally well-written. The method, threat model, and verification protocol are all defined formally and clearly.

### Weaknesses
- The core defense against ambiguity attacks is the dual-passport fidelity gap, for which the paper proposes a formal statistical guarantee using Hoeffding's inequality. However, in the paper's own experiments, this formal guarantee fails for the Mistral-7B model, where the owner's observed gap far exceeds the theoretical threshold.
- The limitations section admits that "An adversary who re-trains on similar data may reproduce the owner's dual entanglement and pass verification by design". The threat model assumes the adversary lacks the fine-tuning data. However, a huge number of popular, shared LoRA adapters are fine-tuned on public datasets. In this very common scenario, an adversary has the model, the (public) SEAL method, and the (public) data. They could simply re-train their own adapter with their own passports, achieving similar performance and creating a valid-looking claim. This severely limits the practical utility of SEAL to only those adapters trained on proprietary datasets.

### Questions
- The failure to achieve the formal statistical guarantee for Mistral-7B (Table 1) is a key concern. Why does the dual-passport gap $\hat{\Delta}_T$ become so large for this specific model? Does this failure correlate with the LoRA rank $r$, model architecture, or task? How sensitive is the gap to these hyperparameters?
- The method seems insecure when the fine-tuning data is public, as an adversary can just replicate the training process with their own passports. Am I misunderstanding the threat model, or does SEAL only offer protection for adapters trained on proprietary datasets? If so, this seems like a major limitation that should be stated more clearly.
- To confirm my understanding of the verification protocol: Does the owner need to privately store the pre-factorized weights $(B, A)$ in addition to the private passport $C_p$ to make a future claim? If so, this implies a 2x storage overhead for the owner's proof, correct?
- How sensitive is the method's performance (both task fidelity and robustness) to the choice of $C$ and $C_p$? What happens if $C$ and $C_p$ are chosen to be (a) (pseudo-)orthogonal, (b) very similar (small $||C - C_p||_F$), or (c) low-rank?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This work proposes SEAL, a universal white-box watermarking scheme for LoRA weights. To address the ownership protection issue of LoRA weights, the scheme inserts non-trainable main passports and auxiliary passports between the trainable matrices of LoRA. Experiments demonstrate that SEAL causes no performance degradation across various tasks and even outperforms standard LoRA in some scenarios.

### Strengths
1.	This paper designs a white-box watermarking mechanism specifically for the LoRA structure to facilitate the ownership protection of relevant weights. 
2.	The method is concise and can maintain model performance across multiple experimental scenarios.

### Weaknesses
1. This work does not explore the impact of matrix properties such as distribution and sparsity on watermark robustness and model performance, nor does it systematically compare the effectiveness differences between different types of passports. The design basis remains insufficient.   

2. The time overhead introduced by this method is still significant (Table 8), and it does not quantify the memory consumption of SEAL during training or the additional overhead during inference compared to standard LoRA.

3. Verification has only been conducted under traditional attack scenarios such as pruning. Can experiments be performed under more advanced attack scenarios currently available?

4. When the rank of LoRA changes, the dimension of the passport matrix needs to be adjusted synchronously. Will this adjustment affect the embedding effect and extraction accuracy of the watermark? Can curves showing the changes in SEAL watermark robustness and model performance under different rank settings be provided?

### Questions
With reference to the Weaknesses section.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
SEAL proposes a white-box watermark for LoRA adapters. During training, two fixed matrices (“passports”) \(C\) and \(C_p\) are inserted between \(B\) and \(A\), and the model alternates them per mini-batch so both versions behave almost identically (they’re “entangled”). After training, \(C\) is factorized and folded into \((B',A')\) so the released adapter looks like a normal LoRA (no extra runtime cost).  
*Public verification* accepts a claim if and only if:
- **R1 (reconstruction):** the submitted passports reproduce the released adapter within a small tolerance,
- **R2 (behavioral gap):** the performance difference between the two passports is below a predeclared threshold \(\tau_T\) (for accuracy metrics, \(\tau_T\) can be set via Hoeffding with a target false-positive rate \(\alpha_T\)).

They evaluate the method on several language, vision-language, and diffusion tasks, and test how well the watermark holds under typical model modifications such as pruning, additional fine-tuning, or re-factorization.

### Strengths
1. **Clear and simple idea.**  
The proposed mechanism is straightforward: alternate two fixed matrices during training and fold one of them into the released adapter. The approach is easy to implement, requires no architectural changes, and adds no inference overhead.

2. **Clearly specified verification rule (but missing one policy detail).**   
The paper precisely describes the public verification procedure through two checks (R1 and R2), and for accuracy-based tasks derives the threshold \(\tau_T\) from Hoeffding’s inequality, giving a clear statistical false-positive rate. However, the current rule still lacks an explicit safeguard against trivial claims (e.g., using identity passports), which should be clarified to make the verification policy complete.

3. **Empirical validation of the verification process.**  
The authors do not just describe R2 but actually test it quantitatively. Across multiple tasks, the measured gaps are small, and cases where the bound does not hold are explicitly marked as “empirical-only” rather than ignored.


4. **Broad evaluation and robustness analysis.**  
The method is tested on several language, vision-language, and diffusion adapters, and under model modifications such as pruning, continued fine-tuning, and re-factorization. The watermark remains detectable under all reasonable perturbations.

5. **Clear presentation and transparency.**   
The paper is generally easy to follow, the decision rule and thresholds are explicitly stated, and an anonymized repository is announced for reproducibility. The authors are transparent about the limits of their guarantees and mark where assumptions may break (e.g., “empirical-only” cases). However, some parts could be improved in terms of clarity and positioning—for instance, the *Spectral Diagnostics* section interrupts the flow—and a short explanation of *why* the theoretical bound fails for the Mistral-7B experiment would make the presentation more complete.

6. **Addresses a real and underexplored need.**  
While white-box watermarking is less practical than black-box verification, focusing on LoRA adapters fills an important gap and could inform more general ownership verification methods in the future.

### Weaknesses
While the paper is well-written and the results are convincing, several aspects could be clarified or strengthened:

1. **Public rule can accept trivial claims unless policy forbids them.**  
   Someone could submit \((B,A)=(B',A')\) and set both passports to the identity matrix. That would pass **R1** (exact reconstruction) and **R2** (zero gap) automatically. If I am not mistaken, the paper does not explicitly say the verifier should reject such submissions or require prior provenance/commitment from the owner. This should be stated clearly in the verification policy.
2. **“Empirical-only” acceptance appears in at least one important case.**  
   For Mistral-7B on the large commonsense suite, the measured gap is above the Hoeffding cutoff, so the formal false-positive guarantee does not apply (acceptance is operational). A short guideline on choosing \(N_T\) and \(\alpha_T\) to keep acceptance formal would help.
3. **Open-data setting weakens the method against determined attackers.**  
   If adapters are trained on fully open-source datasets, an attacker can access the same or very similar data (or sample from the model in the generative case) and *retrain* their own SEAL adapter with correlated passports. With enough compute or queries, this might allow reproducing the behavioral entanglement and pass R2. Even access to a *small* amount of in-distribution data may enable *optimization-based attacks* that search for passports (or for \((B,A,C,C_p)\) tuples) that satisfy R1 and reduce the R2 gap - so the attacker need not always match the defender’s full compute budget. This limitation is worth emphasizing because many LoRA releases rely on public data. However, one has to admit that watermarking always relies on a practical asymmetry: the attacker typically does not want to invest as much compute as the defender—otherwise they could simply train their own model from scratch. 

4. **Placement/clarity of spectral diagnostics.**  
   The *Spectral Diagnostics* subsection is interesting but at the same time confusing, and thus interrupts the flow at the start of experiments. Consider moving/expanding details to/in the appendix or later in the section, and add 1–2 sentences explaining how these diagnostics feed into the main results.

5. **Relevance of white-box watermarking.**  
   White-box watermarking assumes access to the adapter weights and verification via internal parameters, which limits its practical relevance compared to black-box methods. Still, it is a valid and growing research branch that helps understand and strengthen ownership verification under full model access.

### Questions
1. **Attacks using generated or similar data.**  
   The paper mentions that retraining on similar data could, in principle, reproduce the watermark “by design.” Could you expand on how serious this risk is in practice?  
   For example, could an attacker use model-generated samples, a well-trained generative model, or a distillation-style attack to approximate the training distribution and either (a) optimize for new passports that pass R1 and R2 or (b) retrain a SEAL adapter that reproduces the entanglement?  
   A short qualitative discussion of how realistic such data-driven or distillation-based attacks are would help clarify the limits of SEAL’s robustness.

2. **Decision rule corner case / provenance.**  
   As written, Sec. 4.2 (R1 + R2) would accept a trivial claim like \((B,A)=(B',A')\) with \(C_a=C_b=I\). Am I missing something?  
   If not, will you explicitly forbid identical or trivial passports in the public rule to validate ownership?  Please clarify what a verifier *must* demand in practice.

3. **Why the formal FPR bound fails for Mistral-7B.**  
   In Sec. 4.3, the false-positive rate is controlled using Hoeffding’s inequality.  
   For Mistral-7B, the measured gap is larger than the theoretical bound. Could you explain why this happens and which assumption might not be fulfilled in this case?

### Soundness
3

### Presentation
2

### Contribution
3
