---
job_id: b66a19ba-2866-40a4-afad-ee25bb7b42c7
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: 31CznLfRIS.pdf
paper: VideoJudge: Bootstrapping Enables Scalable Supervision of MLLM-as-a-Judge for Video Understanding
main_score_norm: 0.6
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length
Pass ✅.

## Topic Compatibility
Pass ✅.  
The work is on multimodal LLMs, evaluation, and bootstrapped supervision for video understanding, which fits squarely under representation learning, multimodal models, and datasets/benchmarks for ICLR.

## Minimum Quality
Pass ✅.  
All major sections (Abstract, Introduction, Related Work, Methodology, Experiments, Results/Discussion, Limitations, Conclusion) are present and in English. The method is technically coherent, experiments are substantial with multiple benchmarks and ablations, and there are no obvious fatal methodological errors or test leakage issues.

## Prompt Injection and Hidden Manipulation Detection
Pass ✅.  
I do not see any hidden prompts or attempts to steer LLM-based reviewers beyond normal scholarly content.

---

# Expected Review Outcome:

## Summary

The paper introduces VideoJudge, a bootstrapped framework for training multimodal LLM judges specialized for video understanding. A generator–evaluator pipeline produces responses at target quality levels for video–instruction pairs, using iterative refinement and filtering to construct pointwise (1–5 ratings) and pairwise preference data without human labels. Small Qwen2.5-VL based models (3B, 7B) are then fine-tuned as judges, including a variant that generates instance-specific rubrics, and evaluated on several meta-evaluation benchmarks where they match or outperform much larger MLLMs and correlate well with human judgments.

## Strengths

1. **Well thought-out generator–evaluator bootstrapping pipeline.**  
   The methodology in Section 3.1 is carefully specified: Eq. (1)–(4) and Algorithm 1 formalize initial generation, evaluator scoring, and refinement under a deviation threshold \(\alpha\) and max iterations \(T\). The use of detailed video descriptions \(\hat v\) as a proxy for video during bootstrapping is a practical design that materially reduces compute while still tying supervision to video content.

2. **Strong empirical performance relative to substantially larger models.**  
   Table 1 is compelling: VideoJudge-7B reaches Spearman 0.78 / 0.80 on VideoJudgeLLaVA and 0.74 / 0.76 on VideoJudgeVCG, matching or surpassing Qwen2.5-VL-32B/72B, and it keeps solid PSup and \(\Delta(\mathrm{C-D})\) on LongVideoBench where several baselines collapse. Table 3 shows similarly that VideoJudge-3B and -7B reach 94–98.6% accuracy on the VideoJudge pairwise benchmarks and are competitive with or better than Qwen2.5-VL-32B/72B in multiple settings.

3. **Instance-specific rubric generation is an interesting and useful capability.**  
   Section 6.1 and Table 2 show that the rubric-generating VideoJudgeR-3B, trained on only 10% of the pointwise data, attains MAE/RMSE comparable to Qwen2.5-VL-32B/72B and substantially better correlation than its 3B/7B baselines. Figure 3 and Figure 17 further indicate that humans (and GPT-4o-mini as an LLM judge) frequently prefer rubrics from VideoJudgeR-3B over those from much larger models, making the evaluation process more interpretable.

4. **Careful sanity checks on bootstrapped data quality.**  
   Figure 2 shows a clear monotonic degradation in BERTScore and BLEU as the rating gap from the gold response increases, which is a reasonable automatic proxy that the generator is indeed creating progressively worse responses. Figure 16 extends this with VQAScore as a video-grounded metric, again monotonic in the rating, which strengthens the claim that ratings encode semantic quality rather than random noise. Human validation in Table 7 for 2-vs-3 cases demonstrates high inter-annotator agreement and >92% correctness relative to the bootstrap labels, which is nontrivial given that these are the hardest margins.

5. **Useful ablations on temporal context and decoding temperature.**  
   The analysis around Figure 4 and Figure 19 (temperature) and Figure 20 (maxframes) is insightful. It shows that bootstrapped, rubric-trained judges are much more robust to sampling temperature than base Qwen2.5-VL-3B, and it quantifies how many frames are needed during training and evaluation to reach good judgment performance. This is valuable guidance for practitioners deploying MLLM judges in video settings.

6. **Breadth of benchmarks and modalities.**  
   The paper evaluates on multiple meta-evaluation datasets: two bootstrapped suites (VideoJudgeLLaVA/VCG), human-annotated VATEX-Eval, LongVideoBench, VideoAutoArena, and custom human-labeled hard 2-vs-3 pairs. The use of both pointwise metrics (RMSE, MAE, correlations, ECE) and pairwise accuracy gives a well-rounded view of judge quality.

7. **Clarity and reproducibility.**  
   The overall exposition is reasonably clear: Figure 1 is a good high-level visualization of the pipeline, prompts are fully documented in the appendix, and Table 9 lists hyperparameters. The paper promises to release models, datasets, and benchmarks, which would be an important contribution for the community.

## Weaknesses

1. **Closed-loop / self-referential evaluation is not sufficiently interrogated.**  
   The paper acknowledges in Section 7 that both training data and some meta-evaluation benchmarks are constructed via the same generator–evaluator pipeline, but the implications are underplayed.  
   - A substantial part of Table 1 (VideoJudgeLLaVA/VCG) and Table 3 (VJ, VJ-H, at least for non-human parts) measure alignment with labels that originate from Qwen2.5-VL and GPT-4o-mini derived processes. This raises the possibility that VideoJudge is mainly learning to mimic those upstream evaluators rather than human preferences, which could overstate its effectiveness.  
   - The human evaluation components (VATEX-Eval, VideoAutoArena, VJ-H with two annotators) are relatively small compared to the synthetic portions and are not separated out clearly in the headline claims. A more rigorous breakdown of performance strictly on independently human-labeled subsets would make the central claim, “aligns with human judgment,” more convincing.

2. **Heavy dependence on auto-generated video descriptions introduces a potentially serious distributional discrepancy.**  
   The bootstrapping process uses dense descriptions \(\hat v\) in place of raw video for both generator \(G\) and evaluator \(E\), but the deployed VideoJudge is a video model that directly ingests frames at 1 fps (max 60 or 180 frames).  
   - There is no analysis showing that training labels derived from description-based scoring transfer faithfully to frame-based evaluation. For example, descriptions may omit subtle temporal cues that a direct video judge could use, meaning that the judge is trained to respect a projected version of the video rather than the full content.  
   - Eq. (4) and Algorithm 1 use \(\tilde v\) (description) during refinement. If the description has systematic biases (e.g., over-focus on salient objects, missing small events), those biases will be baked into the targets. The paper does not experiment with or discuss the impact of description quality (GPT-4o-mini vs. Qwen2.5-VL-32B) on the resulting judges.

3. **Ground-truth labels largely stem from a single evaluator model, with minimal diversity.**  
   During bootstrapping, a single evaluator \(E\) is used to accept or refine responses (Eq. (2,3)) with threshold \(\alpha\), but the paper does not detail the architecture or training of \(E\), nor whether multiple evaluators are used. This creates a potential “single teacher” bias:  
   - If \(E\) has systematic preferences (e.g., verbosity, stylistic quirks, conservative scoring), these will propagate into VideoJudge, which the meta-evaluation benchmarks, themselves bootstrap-based, may not expose.  
   - The error analysis in Section 6.2 shows large overestimation at high ratings (e.g., 81.3% of rating-4 responses scored as 5), which is quite severe. This bias might originate in the original evaluator and could have been mitigated by mixing evaluators or calibrating to human labels, but that is not attempted.

4. **Limited diversity and coverage of video tasks and domains.**  
   The seed corpus is 25k examples from three instruction-tuning datasets (VideoInstruct-100K, VCG-Plus-112K, VideoChat2-IT), which tend to focus on generic captioning or QA on internet-style videos.  
   - There is no evaluation on more specialized or high-stakes video tasks such as surveillance, medical videos, driving, etc. Table 4 shows the evaluation sets but they are still mainly generic captioning and video QA benchmarks. This limits the claim that VideoJudge is a “scalable evaluator across diverse video understanding tasks.”  
   - Many prompts target fairly simple behaviors (examples in Table 6 are straightforward captioning/summarization). It is unclear how well the judges handle complex temporal reasoning, multi-step logical queries, or tasks where subtle temporal ordering is crucial.

5. **Mathematical formulation and training objectives are somewhat under-specified.**  
   While the NLL training objective in Section 3.2 is standard, several important details are missing or ambiguous:  
   - In the pointwise setting, the target sequence \(t_i\) includes rubric, reasoning, and score tokens, but the paper does not specify whether all tokens are equally weighted or if only the numeric score is used for supervision in some variants. This matters because over-emphasis on natural language reasoning might dilute learning of precise calibration.  
   - Eq. (1) and Eq. (4) assume that conditioning on \(r\) is sufficient for the generator to reliably target a specific rating, but the acceptance threshold \(\alpha\) and maximum iterations \(T\) are never explicitly stated or justified. Given that the error analysis reveals severe overestimation, one wonders whether \(\alpha\) was too loose or \(T\) too small, but no sensitivity analysis is provided.  
   - For pairwise training, the notation overloads \(y_i\) to mean “candidate response (or response pair)” and \(t_i\) to mean “preference label,” but the exact input formatting to the model and whether any ranking loss or derived logit margin is used beyond plain autoregressive token loss is omitted. This makes it harder to reason about the optimization landscape and to reproduce the pairwise models.

6. **Comparisons with prior LLM-as-a-judge work are incomplete, especially in multimodal settings.**  
   While the paper cites some relevant works (e.g., Prometheus(-vision), LLaVA-Critic, Judge Anything), there is little quantitative or qualitative comparison, and some key angles are missing:  
   - The paper does not evaluate against open-source specialized judge models such as Prometheus-Vision or VideoScore/EvQAScore-like systems, even though they are explicitly aimed at multimodal evaluation. This makes it harder to disentangle gains from video-specific bootstrapping versus simply tuning a large model on any judge-like data.  
   - The rubric-generation aspect is conceptually similar to fine-grained evaluator work in text (Prometheus, FLASK), but there is no deep discussion of differences in rubric design or how video-specific challenges are addressed.

7. **Evaluation methodology and statistics could be more rigorous in places.**  
   - For human rubric comparison (Figure 3, Figure 17), the paper reports “win rates” but does not give confidence intervals, nor does it specify how ties or low-agreement items are handled beyond a brief “unanimous vs majority” note. A simple binomial test confidence interval or standard error bars would help interpret the seeming 90%+ win rates.  
   - In Table 3, the “w/ feedback” vs “w/o feedback” configurations are confusing. For some models, feedback hurts (e.g., Qwen2.5-VL-32B on VAA: 90.59 w/o FB vs 80.78 w/ FB), but the paper does not explain what feedback means at inference time for base models or why these reversals occur.  
   - LongVideoBench evaluation treats the judge as scoring correct vs distractor answers and computes PSup and \(\Delta(\mathrm{C-D})\), but there is no statistical significance analysis of the differences between models in Table 1 on this hard benchmark.

8. **Potential reliance on proprietary models and non-open components is under-discussed.**  
   The data pipeline uses GPT-4o-mini for about 33% of video descriptions, and GPT-4o-mini as a rubric judge in some evaluations, but the implications for reproducibility and bias are not deeply analyzed. For instance, if GPT-4o-mini rubrics are themselves referenced as a signal of rubric quality, that introduces another layer of circularity not disentangled from human judgments.

9. **Some examples suggest possible issues with the bootstrapped labels.**  
   Table 6 includes an example under the instruction “What is the man wearing while ironing the dress shirt?” where all R5–R1 responses are about ballet classes in a building, which appears completely mismatched to the instruction. If this is not a typographical error in the paper, it suggests pipeline failures in aligning instruction, video, and responses. The paper does not quantify how often such gross mismatches occur or what filtering, if any, is applied to remove them.

Overall, while none of these issues are fatal, together they temper the strength of the claims about human alignment and generality of the approach. The method is promising, but it would benefit from deeper analysis of bias, teacher dependence, and evaluation rigor.

## Potentially Missing Related Work

Based on the provided context, the following works appear directly relevant and are not cited in the paper:

1. **Chen, X., Li, Y., Wang, Z. (2024), “Multimodal Large Language Models for Video Understanding.”**  
   - *Relevance:* Surveys or analyzes multimodal LLMs specifically for video understanding, which is precisely the application domain of VideoJudge.  
   - *Suggestion:* Discuss in Section 2 (Video Understanding Models and Evaluation) to better position VideoJudge among existing MLLM architectures and training paradigms for video tasks.

2. **Zhang, L., Wu, H., Kim, S. (2023), “Bootstrapping Techniques in Machine Learning: A Comprehensive Survey.”**  
   - *Relevance:* Directly about bootstrapping methodologies in ML, central to the generator–evaluator pipeline.  
   - *Suggestion:* Cite in Section 3.1 when describing the iterative bootstrapping / self-refinement framework, and briefly contrast the proposed pipeline with standard bootstrapping paradigms.

3. **Lee, J., Park, M., Choi, K. (2022), “Evaluating Video Understanding Models: Challenges and Metrics.”**  
   - *Relevance:* Discusses the limitations of classical metrics for video understanding, which is exactly the motivation for using MLLM-as-a-judge.  
   - *Suggestion:* Add to Section 1 and Section 2 to deepen the motivation and relate how VideoJudge addresses the specific shortcomings identified there.

4. **Gonzalez, R., Patel, A., Nguyen, T. (2024), “Synthetic Data Generation for Video Analysis.”**  
   - *Relevance:* Examines synthetic video data and its use in training/evaluating models, closely related to this paper’s synthetic response generation strategy.  
   - *Suggestion:* Reference in Section 3.1 or 5.1 when discussing the pros and cons of using generated responses for supervision.

5. **Huang, Y., Chen, B., Liu, F. (2023), “Interpretable AI in Video Understanding: A Review.”**  
   - *Relevance:* Reviews interpretability approaches in video understanding, directly connected to the rubric-generation claim of improving interpretability.  
   - *Suggestion:* Cite in Sections 1 and 6.1 when arguing that instance-specific rubrics make evaluations more interpretable, and situate rubrics among other interpretability techniques.

6. **Singh, P., Kumar, R., Sharma, L. (2022), “Scalable Supervision in Machine Learning Models.”**  
   - *Relevance:* Discusses scalable supervision, a core theme here (bootstrapped evaluation labels).  
   - *Suggestion:* Reference in Section 3 to contextualize VideoJudge within broader scalable supervision strategies.

7. **Wang, J., Li, S., Zhao, Q. (2023), “Human-in-the-Loop Evaluation for Video Understanding Systems.”**  
   - *Relevance:* Describes human-in-the-loop approaches to video evaluation, which contrast with the fully automated “MLLM-as-a-judge” paradigm.  
   - *Suggestion:* Discuss in Section 2 and the Limitations section as a complementary direction, and clarify scenarios where a human loop might still be desirable despite VideoJudge.

8. **Kim, D., Lee, S., Park, J. (2024), “Advancements in Multimodal Learning for Video Analysis.”**  
   - *Relevance:* Covers recent multimodal learning advances for video, which would help contextualize the choice of Qwen2.5-VL and related baselines.  
   - *Suggestion:* Integrate into Section 2 where video MLLMs are discussed, especially when comparing VideoJudge to off-the-shelf video MLLMs.

9. **Nguyen, H., Tran, P., Le, T. (2023), “Meta-Evaluation Techniques in Machine Learning.”**  
   - *Relevance:* Focuses on meta-evaluation, i.e., evaluating evaluators, exactly the setting of this work.  
   - *Suggestion:* Cite in Section 4.2 when describing the construction and use of VideoJudge meta-evaluation benchmarks.

10. **Rodriguez, M., Chen, L., Wang, Y. (2022), “Challenges in Scaling Video Understanding Models.”**  
    - *Relevance:* Addresses scalability issues in video models, which relates to the motivation for using small 3B/7B judges and to the frame-number ablations in Figure 20.  
    - *Suggestion:* Include in Section 1 or 4.2 when discussing scalability and sample efficiency of small judges versus large models.

## Questions

1. **Diversity of evaluators and alignment with humans:**  
   What is the exact model used as evaluator \(E\) in Eq. (2), and have you tried mixing multiple evaluators (e.g., Qwen2.5-VL-72B + GPT-4o-mini) during bootstrapping to reduce teacher bias? If so, how did it affect overestimation bias and alignment with human judgments on VATEX-Eval and VJ-H?

2. **Description vs. video training mismatch:**  
   Can you provide experiments where judges are trained purely from description-based bootstrapping versus a variant where \(E\) also sees frames (i.e., video in the loop) and compare performance on LongVideoBench and VAA? This would clarify whether the description proxy is sufficient or introduces systematic blind spots.

3. **Sensitivity to \(\alpha\) and \(T\):**  
   What values of \(\alpha\) and \(T\) were used in Algorithm 1, and how sensitive are the resulting judge models to these hyperparameters? If you relaxed or tightened \(\alpha\), does error analysis in Section 6.2 change, especially the severe inflation of mid/high ratings?

4. **Error modes on human-labeled data:**  
   On VJ-H and VATEX-Eval specifically, can you provide qualitative examples where VideoJudge disagrees with humans, and categorize whether errors are due to hallucinations, missing temporal cues, or stylistic preferences? This would help assess whether the judges are safe to use as proxies for humans.

5. **Robustness to adversarial responses:**  
   Did you test judges on adversarial, overly verbose, or misleading responses that exploit known LLM-judge weaknesses (e.g., self-confident reasoning, hedging language)? If not, what failure modes do you expect and how might one harden VideoJudge against such attacks?

Clear answers or additional experiments addressing these points would substantially increase my confidence in the reliability and generalizability of VideoJudge.

## Flag For Ethics Review

- No ethics review needed.

## Details Of Ethics Concerns

N/A. The work uses existing video datasets and automated model judgments for evaluation; there is no direct involvement of human subjects beyond standard annotation tasks and no discussion of deployment in sensitive domains.

## Soundness Rating

3: good.  
The method is technically consistent, the bootstrapping pipeline and training objective make sense, and empirical results across several benchmarks are strong. However, the single-teacher setup, limited analysis of description-vs-video mismatch, and relatively shallow treatment of closed-loop evaluation prevent an “excellent” rating.

## Presentation Rating

3: good.  
The paper is generally clear, with helpful diagrams such as Figure 1 and thorough prompts in the appendix. Some sections (e.g., specification of evaluator \(E\), feedback variants, and hyperparameters like \(\alpha\)) are under-documented, and the occasional example mismatch (Table 6) suggests some sloppiness, but overall readability is solid.

## Contribution Rating

3: good.  
The combination of generator–evaluator bootstrapping for video, small yet strong video judges, and rubric generation is a meaningful contribution to multimodal evaluation. The ideas are not entirely new relative to LLM-as-a-judge in text and images, and reliance on synthetic labels limits theoretical depth, but the empirical results and resources are likely to be useful for the community.

## Overall Rating

6: Marginally above the acceptance threshold. But would not mind if paper is rejected.  

The work provides a solid and practically relevant contribution by adapting LLM-as-a-judge ideas to video understanding with a well-engineered bootstrapping pipeline and careful experiments, and it achieves impressive results with small models. At the same time, there are nontrivial concerns around closed-loop evaluation, teacher bias, and incomplete characterization of failure modes. I lean toward acceptance because the empirical evidence is strong and the released artifacts would be valuable, but the conceptual and methodological issues warrant further scrutiny.

## Reviewer Confidence

4: confident.  
I am familiar with LLM-as-a-judge literature and multimodal evaluation, and I carefully examined the equations, tables (especially Tables 1–3, 7), and figures (1–4, 16–20). Some details about the underlying evaluator and bootstrapping hyperparameters remain unspecified, so there is a small chance I overestimate or underestimate certain risks, but overall I am confident in my assessment.