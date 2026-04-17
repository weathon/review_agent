---
job_id: 65a73740-a9ca-4b2e-80f0-dbbf2964dee7
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: GMlZt4fZSY.pdf
paper: MOBILELLM-R1: Exploring the Limits of Sub-Billion Language Model Reasoners with Open Training Recipes
main_score_norm: 0.8
desk_reject: false
---
# Desk Rejection Assessment:

## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The paper is clearly about data-centric training of small language models and representation learning for reasoning, well within ICLR’s scope.

## Minimum Quality
Pass ✅.  
The paper is complete (abstract, introduction, method, experiments, results, related work, conclusion), written in English, proposes a concrete methodology with substantial experiments, and there are no obvious fatal methodological or statistical flaws.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I did not find any hidden instructions or attempts to manipulate automated reviewing systems in the provided content.

---

# Expected Review Outcome:

## Summary

The paper presents MobileLLM-R1, a family of sub‑billion parameter reasoning language models trained on fully open data with a strong emphasis on data curation and token efficiency. The core technical contribution is a benchmark‑free, influence‑based data selection and mixing pipeline for both pre‑training and mid‑training, including leave‑one‑out analysis and self‑evolving data‑model co‑optimization. With only 4.2T pre‑training tokens, the 950M model matches or surpasses Qwen3‑0.6B and substantially outperforms fully open models like OLMo‑2 and SmolLM‑2 on a range of reasoning benchmarks, and the paper releases models, code, and detailed recipes.

## Strengths

1. **Clear, end‑to‑end recipe with strong open baselines and competitive results.**  
   The paper provides a fully specified training pipeline from scratch pre‑training through mid‑training and SFT, including hyperparameters (Table 4), architecture (Table 3), and detailed data mixtures (Tables 5–7). Empirically, MobileLLM‑R1‑950M is very competitive: Table 8 shows 61.6 GSM8K (8‑shot) and 46.3 HumanEval, beating all fully‑open base models below 2B parameters and even surpassing Qwen3‑0.6B on code. Table 9 further shows that after reasoning SFT, the 950M model reaches 15.5 AIME’24 and 19.9 LCBv6, outperforming all fully open baselines and approaching much larger partially open models. This level of detail plus strong baselines makes the work quite impactful and practically useful.

2. **Data‑centric methodology that is more principled than typical heuristic mixing.**  
   The paper goes beyond “just another LLM” by focusing on *how* to use a constrained token budget effectively. The leave‑one‑out (LOO) group impact analysis (Eq. (1), Figure 3) quantifies per‑dataset utility on capability‑probing sets, revealing non‑obvious interactions (e.g., StarCoder helping math more than OpenWebMath helps code). Building on AutoMixer‑style influence scores (Eq. (2)–(5)), the authors derive dataset weights that jointly account for self‑ and cross‑capability influence, rather than heuristic mixing. Figure 4 shows that this derived “Datamix” consistently yields lower perplexity than uniform sampling on unseen reasoning benchmarks across math, code, and knowledge, lending credibility to the benchmark‑free optimization claim.

3. **Self‑evolving mid‑training with influence‑based compression is interesting and empirically justified.**  
   Section 3 introduces an iterative mid‑training scheme where influence scores with respect to capability‑probing sets are used for (i) sample‑level rejection (Eq. (6)) and (ii) dataset‑level reweighting (via Eq. (4)–(5)), with the model and data co‑evolving. Figure 5’s histograms clearly show the “compression” phenomenon: from Stage 1 to Stage 2, influence distributions for both general knowledge and math become narrow and centered near zero, consistent with the claim that informative samples are being exhausted. Figure 6 (described in text on Page 7) shows that the subsampled mid‑training data avoids the pronounced performance dip around 30K steps that occurs with the original data, both under cross‑entropy and KD, which strongly supports the value of this compression scheme.

4. **Careful experimental controls to disentangle pre/mid‑training from SFT.**  
   Table 2 is particularly informative: all models (SmolLM2, OLMo‑2, MobileLLM‑R1) are fine‑tuned on *the same* reasoning SFT mixture. Under this controlled setup, the MobileLLM‑R1 models still dominate, e.g., the 359M variant achieves 19.2 MATH vs. 5.2 for SmolLM2‑360M and 23.8 GSM8K vs. 7.4. At the ~1B scale, the 950M variant surpasses OLMo‑2‑1.48B (57.8 vs. 53.0 MATH, 68.5 vs. 58.8 GSM8K). This is strong evidence that the gains are not merely from better SFT data but from the data‑centric pre/mid‑training strategy.

5. **Insightful analyses beyond raw scores.**  
   The paper includes several nice diagnostic studies. For example, Section D.1 explores how pre‑training learning rate affects representation quality: Table 11 shows that higher LR leads to higher RankMe scores (Eq. (8)) and better post–mid‑training MMLU, even though pre‑training MMLU is similar. This is a non‑obvious and useful insight for practitioners. Section D.2’s RL ablation (Figure 10) also adds nuance: small models can benefit from RL when starting from a base checkpoint, but for an SFT‑optimized small model, extra RL may degrade GSM8K, supporting the paper’s emphasis on SFT with curated traces over RL on small models.

6. **Figures effectively support the narrative.**  
   Several figures are well chosen and informative:  
   - **Figure 1** plots HumanEval accuracy vs approximate pre‑training FLOPs and shows MobileLLM‑R1 lying on a very favorable Pareto frontier compared to Qwen and SmolLM/OLMo, visually substantiating the “token‑efficient accuracy” claim.  
   - **Figure 7** tracks perplexity on GSM8K and HumanEval across phases, showing a striking drop in GSM8K PPL during the second pre‑training phase and a later drop in code PPL during mid‑training, nicely illustrating the claimed math‑to‑code transfer.  
   - **Figures 8 and 9** summarize base and post‑trained model comparisons across multiple benchmarks; in particular, Figure 9 highlights that MobileLLM‑R1‑360M is uniquely strong on LiveCodeBench vs models up to 1.7B parameters.

7. **Reproducibility and openness.**  
   The paper promises release of all models and code, and the training recipe (Tables 3–7, Section A.2) plus the explicit data sources and ratios are unusually detailed for this line of work. Given that most strong reasoning models today rely on opaque proprietary corpora, this is a valuable contribution for the community.

## Weaknesses

1. **Reliance on approximate influence estimation without sufficient sensitivity analysis.**  
   The central technical mechanism (Section 2.2 and Section 3) leans heavily on AutoMixer‑style influence scores (Eq. (2)), which depend on approximating the inverse Hessian and using a small number of checkpoints. While AutoMixer is cited, the paper effectively reuses its methodology at much larger scale and in a quite different setting (multi‑domain decoder LMs). The authors compute joint influence (Eq. (4)) and then average over representative subsets (Eq. (5)), but there is little quantitative evaluation of how robust the resulting weights \(w_g\) are to choices like the number of checkpoints \(T\), the linear weighting \(\alpha_{c,t} \propto t\), or the size / composition of representative datasets \(\mathcal{D}^R_i\). For example, there is no ablation that recomputes Datamix with fewer checkpoints or different \(\alpha_{c,t}\) to show the resulting mixture and downstream accuracy are stable. Given how noisy influence estimates can be, this is an important missing piece for trusting the method as a general recipe rather than a one‑off engineering success.

2. **Some mathematical definitions are under‑specified or raise identifiability questions.**  
   A few formulations that are central to the method are only sketched at a high level:
   - In Eq. (1), the group impact \(\Delta \mathcal{L}(\mathcal{D}_j, \mathcal{D}^{\mathcal{P}}_{\mathcal{C},\mathcal{M},\mathcal{K}})\) is defined using models \(\hat\theta\) and \(\hat\theta_{-j}\), but it is not stated whether these models are trained to equal total steps or equal total tokens after removal, and how learning rate schedules are adjusted. This matters because the NLL trajectories in **Figure 3** could partly reflect different effective training budgets rather than only the marginal utility of the omitted dataset.  
   - In Eq. (5), the dataset‑level score \(\rho_g\) normalizes by \(N_g\) (token count) and multiplies by sequence length \(s_i\). This effectively weights each example proportional to its length relative to \(N_g\), but it is unclear why this is preferable to per‑token averaging or whether it biases toward longer documents. A more explicit derivation or justification would help.  
   - For mid‑training compression (Eq. (6)), the threshold is simply \(I(x_i;\theta_t) > 0\). Influence estimates are noisy; some kind of margin or smoothing would be expected. There is no discussion of how many points are borderline or how sensitive the final dataset is to small perturbations of this threshold.

3. **Scope of “benchmark‑free” claim is overstated.**  
   While the authors emphasize that no held‑out benchmarks are used for data mixing, capability‑probing datasets (Section 2.1.1) are derived via a pipeline that uses (i) FineWeb‑Edu scores, (ii) Ask‑LLM judgments, and (iii) domain‑specific prompts that are clearly tailored to math/code/knowledge distributions. Moreover, **Table 6** explicitly includes benchmark training sets (GSM8K, ARC, OBQA, etc.) in mid‑training. This is not wrong per se, but the paper sometimes suggests that the mixture is optimized in a fully benchmark‑agnostic way. In practice, capability‑probing sets and benchmark training sets are close in style to the evaluation benchmarks; the claim should be softened or more carefully delimited.

4. **Evaluation breadth is still somewhat narrow given the strong claims.**  
   The core empirical story is about math, code, and some general knowledge. However, the base model evaluation in Table 8 focuses on MATH500, GSM8K, MBPP, HumanEval, a commonsense average, and MMLU. The post‑trained evaluation (Table 9) focuses on MATH500, GSM8K, AIME’24/25, and LCBv6. These are all reasoning‑heavy, which is aligned with the paper’s goals, but there is limited evidence on broader language understanding, generative quality, or robustness (e.g., instruction following beyond Tülu tasks, safety, multilingual, long‑document QA). In particular, **Figure 8** shows that MobileLLM‑R1‑360M has worse MMLU than SmolLM2‑360M despite dramatically better GSM8K and HumanEval; similarly, in Table 1, adding math and code reasoning SFT often degrades MMLU. This suggests a fairly sharp specialization; the paper’s narrative would benefit from a more explicit discussion and perhaps additional metrics to quantify this trade‑off.

5. **Limited ablations on key design choices in the capability‑probing pipeline.**  
   The hierarchical rejection sampling (Section 2.1.1) uses FineWeb‑Edu score > 4, top‑10% Ask‑LLM probabilities, and semantic deduplication to create representative sets of about 10k examples per dataset. These small sets then drive both the LOO analysis (**Figure 3**) and influence estimation. However, the paper provides no evidence that these specific hyperparameters (e.g., 10% top cutoff, 10k size) are near‑optimal or even not brittle. It would be very informative to see, for a single capability, what happens if the probing set size is halved or the Ask‑LLM score quantile is changed, and how much the final Datamix performance in **Figure 4** varies. Without this, it is hard to know whether practitioners can safely follow the procedure or need to redo extensive tuning.

6. **Computational cost vs. benefit trade‑off is only roughly discussed.**  
   Section D.3 gives a back‑of‑the‑envelope cost comparison: ~6.8k GPU hours for curation (influence + LOO + sampling) versus ~38.6k GPU hours for model training. While this is “only” about 15–20%, it is still a substantial absolute cost, particularly for smaller labs, and much of it is incurred before any model is fully trained. There is no ablation showing, for example, how a simpler heuristic scheme (e.g., manual reweighting based on LOO curves in **Figure 3** but no full influence‑based Datamix) would fare as a cheaper approximation. Given that the main selling point is token *efficiency*, it would be useful to understand whether the method improves *overall compute efficiency* versus a naive 6T‑token pretraining with uniform mixing and no influence estimation.

7. **Some comparisons and claims of superiority need more nuance.**  
   The paper frequently contrasts its 4.2T token budget with Qwen3’s 36T and states that MobileLLM‑R1‑950M “matches or surpasses Qwen3‑0.6B across multiple reasoning benchmarks” (Abstract). However, Table 8 shows Qwen3‑0.6B‑Base at 29.8 MATH500 vs 26.8 for MobileLLM‑R1‑950M (4‑shot) and 60.9 vs 61.6 GSM8K, and Qwen3 slightly ahead on MMLU. Table 9 shows that after instruction tuning, Qwen3‑0.6B is still somewhat stronger on GSM8K (79.2 vs 67.5). The main clear advantage is LCBv6 (19.9 vs 14.9) and code (HumanEval 46.3 vs 30.5). The text should more carefully distinguish “comparable overall, clearly better on code and some reasoning metrics” from “matches or surpasses” in aggregate.

8. **RL ablation is interesting but under‑instrumented.**  
   The RL study in Section D.2 and **Figure 10** uses GRPO on NuminaMath‑TIR with \(\beta=0\) and 100 steps. It is not clear whether this is a reasonably strong RL setting or a minimal toy experiment. Given the current interest in RL‑based reasoning (e.g., DeepSeek‑R1, Qwen3), this section could easily be misread as “RL does not help small models” when in fact the experiment is very narrow (single dataset, single schedule, small number of updates). Some clarification and positioning would avoid over‑interpretation.

9. **Minor clarity issues and missing citations to canonical pretraining work.**  
   - The related work section focuses on recent LLMs and reasoning models, but the paper does not cite several canonical parameter‑efficient or transfer‑learning–oriented LMs (T5, BERT, ELECTRA, ALBERT, DistilBERT, GPT‑2) that helped define the landscape of pretraining and model compression; acknowledging these would better contextualize MobileLLM‑R1 in the broader sequence of “small but capable” models.  
   - Notation in Eq. (1)–(5) occasionally reuses symbols (\(\mathcal{D}^{\mathcal{R}}_i\), \(\mathcal{D}^{\mathcal{P}}_c\), \(N_g\)) in a way that forces the reader to back‑track to Section 2.1.1; a compact notation table would help.

Overall, the weaknesses are mostly about the breadth and robustness of the methodology rather than clear errors; they do not undermine the empirical contribution but they do limit how “principled” and general the proposed pipeline feels.

## Potentially Missing Related Work

1. **Raffel et al., “Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (T5)”, 2020.**  
   This work is a foundational study on large‑scale pretraining and transfer with a focus on data mixture, scaling, and efficiency. It should be cited in Section 5 when discussing scaling laws and data‑centric training, and possibly in Section 2 as a contrast between their largely uniform pretraining and the influence‑based Datamix used here.

2. **Devlin et al., “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding”, 2019.**  
   BERT is a core reference for pre‑training corpora and data quality considerations. It would provide important context in Related Work for how earlier models leveraged Wikipedia + BookCorpus vs the more curated fine‑web and math/code mixtures here.

3. **Radford et al., “Language Models are Unsupervised Multitask Learners (GPT‑2)”, 2019.**  
   GPT‑2 is the canonical decoder‑only LLM that first highlighted broad zero‑shot multitask abilities, making it directly relevant to this work’s focus on small decoder LMs. It should be cited in the Introduction or Related Work when positioning MobileLLM‑R1 among decoder‑only models.

4. **Clark et al., “ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators”, 2020.**  
   ELECTRA presents a more sample‑efficient pretraining objective. While MobileLLM‑R1 focuses on the standard generative objective, ELECTRA is relevant to the broader question of “doing more with fewer tokens” and should be discussed as an alternative axis of efficiency in Section 5.

5. **Lan et al., “ALBERT: A Lite BERT for Self-supervised Learning of Language Representations”, 2020.**  
   ALBERT is a key parameter‑reduced LM that balances capacity and efficiency by sharing parameters, similar in spirit to MobileLLM’s weight tying. It would fit naturally into the Related Work paragraph on small models like MobileLLM, OLMo, SmolLM.

6. **Sanh et al., “DistilBERT, a Distilled Version of BERT: Smaller, Faster, Cheaper and Lighter”, 2019.**  
   DistilBERT is an early distillation approach to building compact LMs. Given that this work emphasizes on‑device, sub‑billion models, DistilBERT should be mentioned in Related Work as a precursor to current small language models.

(Other papers in the provided list such as He et al. ResNet, Zoph et al. NAS, and Joulin et al. Bag‑of‑Tricks are only tangentially related and do not need to be cited.)

## Questions

1. **Robustness of influence‑based Datamix.**  
   Could the authors provide any evidence (even in the rebuttal) on how sensitive the Datamix in Eq. (5) is to the choice of checkpoints \(T\) and weights \(\alpha_{c,t}\)? For example, if you use only the last 3 checkpoints or uniform \(\alpha_{c,t}\), how much does the resulting mix and final GSM8K/HumanEval performance change?

2. **Thresholding for mid‑training compression.**  
   For Eq. (6), what fraction of \(\mathcal{D}^{\text{(raw)}}\) survives in Stage 1 and Stage 2 respectively? Did you experiment with requiring \(I(x_i;\theta_t) > \tau\) for some positive margin \(\tau\) instead of just > 0? Some statistics (e.g., ROC‑like curves relating \(\tau\) to retained tokens and performance) would clarify how brittle this step is.

3. **Alternative cheaper approximations to the full data curation pipeline.**  
   Given the ~6.8k GPU‑h overhead for influence + LOO, did you try any simpler heuristics like (i) only LOO curves from Figure 3, (ii) a small set of capability‑specific losses without per‑sample influence, or (iii) manual upweighting of FineWeb‑Edu and math/code sets? It would be helpful to know whether 80% of the gains can be captured with, say, 20% of the extra compute, for researchers with smaller budgets.

4. **Benchmark trade‑offs and generality.**  
   Can the authors clarify whether there are tasks where the MobileLLM‑R1 base models *underperform* SmolLM or OLMo beyond MMLU (e.g., more open‑ended generation, summarization, or dialogue)? If such results exist, they would help characterize the specialization trade‑off and prevent over‑generalization of the claims.

5. **RL configuration details.**  
   For the GRPO experiments in Figure 10, can you justify the chosen hyperparameters (learning rate, number of steps, \(\beta=0\)) as reasonably strong baselines? Have you tried any variants with non‑zero KL penalties or longer training to ensure the conclusion “SFT > RL for small models” is not an artifact of a weak RL setup?

Answers to these questions, especially quantitative sensitivity experiments around influence‑based curation, could significantly increase my confidence in the generality of the proposed methodology.

## Flag For Ethics Review

- No ethics review needed.  

## Details Of Ethics Concerns

N/A.

## Soundness Rating

3: good.  
The technical approach is mostly sound and the empirical support is strong, but key methodological components (influence estimation, probing set construction) are not deeply stress‑tested for robustness.

## Presentation Rating

3: good.  
The paper is generally well written, with clear structure and informative figures and tables, though some notation is dense and the benchmark‑free claim could be more carefully qualified.

## Contribution Rating

3: good.  
The work offers a meaningful contribution in demonstrating strong reasoning in sub‑billion models with a transparent, data‑centric recipe, and introduces a reasonably principled influence‑based data mixing and compression pipeline, even though some aspects feel more engineering‑driven than fully theoretically grounded.

## Overall Rating

8: Accept, good paper (poster).  
Despite some missing robustness analyses and slightly overstated framing, this is a solid, carefully executed piece of work with strong empirical results, an unusually open and detailed recipe, and a data‑centric methodology that will be valuable to the community.

## Reviewer Confidence

4: confident.  
I am familiar with LLM pretraining, data curation, and influence‑based methods, and have carefully examined the equations and experimental tables, though I have not independently reimplemented AutoMixer‑style influence estimation at this scale.