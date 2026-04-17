# A Little Help Goes a Long Way: Efficient LLM Training by Leveraging Small LMs

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
A primary challenge in developing large language models (LLMs) is their onerous pre-training cost. This paper explores a promising paradigm to improve LLM pre-training efficiency and quality by leveraging a small language model (SLM). In particular, this paradigm relies on an SLM to both (1) provide soft labels as additional supervision, and (2) select a small subset of valuable training examples. Put together, this enables an effective transfer of the SLM's predictive distribution to the LLM, while prioritizing specific regions of the training data distribution. Empirically, this leads to reduced LLM training time compared to standard training, while improving the overall quality. Theoretically, we develop a statistical framework to study the utility of SLMs in enabling efficient training of high-quality LLMs. Our framework characterizes how the SLM's seemingly low-quality supervision can enhance the training of a much more capable LLM. Furthermore, it also highlights the need for an adaptive utilization of such supervision, by striking a balance between the bias and variance introduced by the SLM-provided soft labels. We corroborate our theoretical framework by improving the pre-training of LLMs with 2.8B and 8.6B parameters by utilizing smaller LMs on the Pile dataset.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes SALT (Small model Aided Large model Training), a two-stage pre-training method for Large Language Models (LLMs). It leverages a Small Language Model (SLM) as a teacher for Knowledge Distillation (KD) in the initial stage, followed by standard pre-training. An optional extension $SALT_{DS}$ uses the SLM to select "challenging yet learnable" data for the KD phase. The authors provide a theoretical analysis presenting risk bounds for KD in the language modeling context. Empirically, SALT is shown to achieve performance comparable to or better than a standard baseline with fewer training steps (70%), resulting in significant wall-clock time savings (25-29%) for 2.8B and 8.6B models, and also improves performance after SFT.

### Strengths
The paper tackles the critical and practical problem of reducing the high computational cost of LLM pre-training.

It demonstrates significant wall-clock time savings (25-29%) on large-scale models (2.8B, 8.6B) while maintaining or improving performance over the baseline.

The paper provides a theoretical analysis by developing risk bounds for knowledge distillation specifically in the autoregressive language modeling context.

The improvements from SALT pre-training are shown to carry over to downstream tasks after supervised fine-tuning.

### Weaknesses
The core components (small-to-large KD, early-stage KD, data selection via small models) are not new. The claimed novelty rests on a specific combination of these ideas, and the theoretical contribution appears to be an application of existing analyses to this specific setting.

The paper lacks direct empirical comparisons against stronger, contemporary baselines for both data selection and KD scheduling. It is also unclear if the standard baseline was sufficiently tuned.

### Questions
Can you clarify the specific novel contributions over prior work (like Qin et al., 2022, Ankner et al. 2024) beyond scale and the specific data selection heuristic?

The method performed poorly with a 0.5B teacher. What is the explanation for this performance degradation, and what does it imply for the method's practical limits?

If training step is longer, does the improvement disappear?

If training data is more recent clean one like fineweb-edu, does the improvement disappear?

How about the cost of hyper parameter search? And, generalization toward reuse of the selected hyperparameters to other settings?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper addresses a core problem in the large language model (LLM) field: high pre-training costs. The authors propose a novel and efficient training paradigm, leveraging small language models (SLMs) to aid the training of LLMs, which they name SALT (Small model aided large model training).

The core contributions of this method are twofold:
1.  **Theoretical Framework:** The paper first establishes a statistical framework to theoretically analyze the application of knowledge distillation (KD) in language models. Specifically, it explores the feasibility of "reverse distillation" (using a *weaker* SLM as a teacher to train a *stronger* LLM student). The theory reveals this as a **bias-variance trade-off**: soft labels from the SLM can reduce training variance, but due to its weaker capability, it also introduces bias (especially on "hard" samples).
2.  **SALT Algorithm:** Guided by this theory, the authors designed the SALT algorithm. It is a **two-stage training method**:
    * **Stage 1 (KD):** In the early phase of training ($n_{KD}$ steps), it uses the SLM as a teacher for knowledge distillation, capitalizing on its low bias and variance-reduction benefits in "easy" data regions.
    * **Stage 2 (Standard):** In the subsequent phase, it switches back to standard (ground-truth-based) next-token prediction training, allowing the LLM to learn the "hard" samples that the SLM could not master.
3.  **$SALT_{DS}$ (Data Selection):** The paper further proposes an extension, $SALT_{DS}$, where the SLM is also used for **data selection**. It uses a scoring function (Eq. 10) to filter for "**challenging yet learnable**" training sequences, specifically for the KD in the first stage.

**Experimental Results:**
The authors validate their method by training 2.8B and 8.6B parameter LLMs on the Pile dataset (using 1.5B and 2.8B SLMs as teachers).
* **Efficiency:** SALT-trained LLMs can match (or exceed) the performance of standard-trained (BASELINE) LLMs using **less than 70% of the training steps**.
* **Performance:** The final SALT models outperform the BASELINE on a wide range of few-shot benchmarks and supervised fine-tuning (SFT) tasks.
* **Time Savings:** This translates to an estimated **~25% (2.8B) to ~28% (8.6B) wall-clock time saving**.

### Strengths
1.  **Addresses a Critical Problem:** The work focuses on reducing LLM pre-training costs, which is a highly important and valuable research direction.
2.  **Reported Efficiency Gains:** The paper reports significant wall-clock time savings (~25-28%). If robust, this result is of high practical value.
3.  **Theoretical Framework:** The paper provides a theoretical framework that attempts to explain the "weak-teacher-strong-student" distillation mechanism via a bias-variance trade-off.

### Weaknesses
1.  **Unclear and Minimal Contribution of $SALT_{DS}$:** This is a **major weakness**. The paper introduces $SALT_{DS}$ (with data selection) as a key extension, yet the experimental results show it offers **no significant or consistent benefit** over the simpler SALT baseline (e.g., 8.6B model avg @100% steps: SALT 52.96 vs $SALT_{DS}$ 52.81). This makes the contribution of the data selection part *nearly void*. The authors need to justify why this more complex method (requiring an extra SLM scoring pass) is proposed if it provides no tangible benefit.
2.  **Severe Lack of Ablation for Data Selection:** The core of $SALT_{DS}$ is the scoring function in Eq. (10), which relies on a critical hyperparameter $k$ (set to 10) to define "learnability". The paper provides **no ablation or sensitivity analysis** on this $k$ value. What happens if $k=1$ or $k=50$? Without this analysis, the effectiveness of this data selection mechanism is **unproven**, and its design appears arbitrary.
3.  **Unprincipled Choice of $n_{KD}$:** The choice of the transition point $n_{KD}$ (36K steps) seems **arbitrary**. While Appendix K shows robustness for $n_{KD}$ between 20k and 60k, the paper fails to discuss how one might *principally* determine this optimal transition point. Lacking this discussion makes the method feel more like a specific "recipe" than a general approach.

### Questions
1.  **($SALT$ vs $SALT_{DS}$) Necessity of $SALT_{DS}$:** As noted in the weaknesses, the performance of $SALT$ and $SALT_{DS}$ is very close. Can the authors elaborate on whether the practical benefit of $SALT_{DS}$ justifies its additional complexity (i.e., a full forward pass over the pre-training data by the SLM)? Under what conditions would you expect $SALT_{DS}$ to *significantly* outperform $SALT$?
2.  **($SALT_{DS}$) Data Selection Hyperparameter $k$:** The choice of $k$ in Eq. (10) seems critical. You used $k=10$. Did you experiment with other $k$ values? For example, how would performance be affected by a very small $k$ (e.g., $k=1$, focusing only on tokens the SLM gets right) or a very large $k$ (e.g., $k=100$)? This is important for understanding the definition of "learnability."
3.  **(SALT) Choice of Transition Point $n_{KD}$:** The ablation in Appendix K (Table 22) shows $n_{KD}=60K$ has slightly better average performance (47.99) than $n_{KD}=36K$ (47.94). You mention choosing 36K for efficiency. My question is: is there a *principled* way to determine the optimal $n_{KD}$? For example, does it correspond to a point where the SLM teacher's training loss begins to "saturate," or some metric indicating that "easy" samples have been sufficiently learned?

### Soundness
3

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
3

### Summary
The authors propose SALT (Small‑model Aided Large‑model Training), a two‑stage pre‑training recipe: early knowledge distillation (KD) from a smaller teacher followed by standard next‑token training; an extended variant (SALTDS) uses the SLM to select “challenging yet learnable” sequences for the KD phase (Algorithm 1, p. 5). A statistical framework (Theorems 3.2 & 3.4) provides excess risk bounds for language modeling under KD, highlighting a bias–variance trade‑off where SLM‑provided soft labels can reduce variance if teacher–data divergence remains small (pp. 3–4). Empirically, 2.8B/8.6B students pre‑trained on The Pile with UL2 show that SALT/SALTDS reach or exceed baseline few‑shot performance using 70% of training steps and achieve ~25–29% wall‑clock savings, with further downstream gains after supervised fine‑tuning (Tables 1–4, pp. 7–8; Appendix J, p. 44). A teacher‑size ablation (Table 3, p. 8) shows benefits diminish when the teacher is very small (0.5B).

### Strengths
- Clear, simple method with practical payoff. SALT’s two‑stage schedule is easy to implement and consistently improves overall few‑shot averages at the same step budget, and matches/beat baselines at 70% steps, delivering ~25–29% time savings in the authors’ TPU setup (Table 2 Table 13).

- Theory tailored to language modeling under KD. The paper gives sequence‑level generalization bounds for KD in LMs (Theorem 3.4), relating gains to reduced variance vs teacher–data divergence, and motivates selective/early KD (Remark 3.5 & §3.3). 

- Thoughtful diagnostics & positioning. The histograms of  (Fig. 2; Appendix C) empirically support the stability assumption; the bucketed analyses (e.g., Table 5 on XLSum‑EN,; Appendix M) illustrate that early KD mostly helps “easy” slices, which fits the theory and design.

- Robustness checks. Ablations on transition strategies and KD duration and on teacher quality (Table 21) give a fuller picture; the method still helps across two student sizes (2.8B & 8.6B).

### Weaknesses
1. Fairness of the efficiency claim at 70% steps. The core headline—“SALT surpasses BASELINE at 146k (~70%) steps” (Table 2 Table 13)—is not matched with a BASELINE@70% few‑shot evaluation. The only “early” baseline shown near the main text is BASELINE@36k (Table 1 & Fig. 3), which is too early to compare to 146k. Without BASELINE@146k, it’s hard to attribute step‑efficiency to KD rather than general training dynamics. Please include BASELINE evaluated at identical step counts reported for SALT/SALTDS.

2. Theory–practice gap in key assumptions. The token‑level bound (Theorem 3.4) assumes a finite function class 
Θ and bounded per‑token log‑loss (Assumption 3.1), and quantifies bias via TV distance to the data conditional distribution—a quantity that’s intractable to estimate in practice. The empirical proxy uses completions from a strong LM “oracle”, which approximates model‑to‑model divergence, not teacher‑to‑data divergence; this weakens the validation of DIV Clarify how the bounds should be interpreted operationally given these gaps.

3. Baselines could be stronger. Reverse KD (RKD) is a useful strawman but not a competitive baseline. Consider curriculum KD top‑k token KD (Appendix A.2, p. 19) with tuned  𝑘 or self‑distillation controls. For SALTDS, a data‑selection baseline that mimics the same selected subset without KD would isolate the data‑selection contribution more fairly.

4. Statistical reporting. Few‑shot results are single‑run with no seeds/variance/confidence intervals. Given small deltas (e. g., +0.62 average points for 2.8B at 100% steps; Table 2), error bars are important.

5. Cost accounting transparency. Wall‑clock savings (25–29%) depend on specific hardware and rematerialization settings. Reporting FLOPs or a normalized throughput‑adjusted cost would make claims more portable.

6. Scope of evaluation. The Pile is English‑heavy; a brief multilingual check or domain‑shift test (e.g., code/math) beyond MBPP and MATH citations would strengthen generality claims. (You do show strong LAMBADA gains—Table 2—but broader coverage would help.)

### Questions
1. BASELINE@146k: Can you report few‑shot averages and domain breakdowns for BASELINE at 146k steps (2.8B and 8.6B) to match the SALT reporting? This is critical for the step‑efficiency claim.

2. SALTDS selection: In Eq. (10) (p. 6), you use median of filtered per‑token losses with top‑k masking. Did you try mean/trimmed‑mean or per‑document entropy to balance “challenging yet learnable”? Any diversity constraint to avoid duplicative sequences?

## Suggestions

- Add BASELINE@70% few‑shot (and maybe @80%, @90%) to align curves across methods. Also show SALT@70% vs BASELINE@70% on key individual tasks (beyond overall averages)

- Report variability. Provide ≥3 seeds for the few‑shot suite with mean±std or 95% CIs; likewise for SFT results

- Stronger baselines. Include self‑distillation and top‑k token KD; try KD weight annealing beyond the tested linear variants (Appendix K, Table 23)

- Wider evaluation. Include multilingual (e.g., TyDi beyond English) or code/math reasoning benchmarks, and a brief data‑shift analysis to test whether SALT’s gains persist out of distribution.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper studies the use of pretrained smaller language model (SLM) to train larger ones via knowledge distillation (KD).
The paper first provide theoretical studies of the risk bound including the effects of KD.
Then, it proposes a two-stage approach (SALT) of pretraining reusing SLM: First, perform KD to train the model (optionally select learnable data to further accelerate training). Then, pretrain the model as usual.
This utilize the intuition that SLM can help learning easier tasks, and at the latter stage the capacity of the larger model can be leveraged to further train itself.

The paper provides experimental validation comparing mainly with from-scratch pretraining and KD (without second stage). Results for post-training are also given.

### Strengths
- The paper provides both theoretical and empirical contributions to KD with smaller models.
- Experiments are performed with care, with ablation studies given to justify the hyperparameters used, etc.

### Weaknesses
1. Data selection method is not so practical as it involves using early model checkpoint, which is not typically available if one wishes to use off the shelf public models. Moreover, it involves more hyper parameters for tuning.
2. Lack of baseline: I think the proposed method should be compared to model growth methods as baselines. These methods are quite straightforward compared to SALT (simply stacking or expanding the widths works well; [2405.15319]) and are also reusing small models to accelerate the training of large ones.
3. While the theory provides some clue on why KD helps, it does not provide concrete guidance on when or how to transition from KD to pretraining without it. Using a two stage training process works well intuitively without the theory as well (weak teacher can provide supervision to a strong but blank-state student only in the beginning), making the theoretical contribution seemingly redundant.

### Questions
1. The accuracies seem to be converging with larger train steps. I wonder if at even larger train steps from-scratch pretraining becomes better and KD becomes unnecessary?
2. The improvement seems to be small compared to from-scratch pretraining. I wonder, when both baseline and KD are compared under equal compute (KD requires extra compute due to the inference of SLM), from-scratch pretraining could perform better?

### Soundness
3

### Presentation
3

### Contribution
3
