=== CALIBRATION EXAMPLE 60 ===

# Harsh Critic Review
## Section-by-Section Critical Review

**Title & Abstract:**
The title, "DeepCritic: DELIBERATE CRITIQUE WITH LARGE LANGUAGE MODELS," is apt and clearly indicates the paper's focus. The abstract succinctly summarizes the problem, the two-stage solution (SFT on curated deliberate critiques and RL), and the key results: superior performance to existing critics (including GPT-4o) on error identification and more effective refinement feedback. The claims are strong and set high expectations for the experimental section. One minor note: the abstract mentions "4.5K long-form critiques," but the main text (Section 3.2.1) clarifies this as "approximately 4.5K seed solution-level critiques." The distinction is minor but important for precision.

**Introduction & Motivation:**
The motivation is well-articulated and aligns with important problems in the field (scalable oversight, improving LLM critique quality). The core issue—that existing LLM critics produce shallow, non-critical critiques—is clearly established. The introduction successfully frames the work within the context of prior research on LLM critics and reasoning. The contributions are listed implicitly but are clear from the narrative: a two-stage training pipeline, a method for curating "deliberate" critique data, and extensive evaluation showing gains in accuracy and feedback utility.

**Methodology:**
The two-stage pipeline is described in detail. The use of step-wise initial critique generation, followed by in-depth critique (multi-perspective verification and meta-critiquing), and then synthesis is a novel and well-justified process for generating high-quality SFT data.
*   **Reproducibility & Assumptions:** The method is largely reproducible. Prompts are said to be in Appendix A (though not visible in the provided text). A key assumption is that the teacher model (Qwen2.5-72B-Instruct) is capable of generating high-quality in-depth critiques. The authors mitigate this by filtering: they only keep samples where the in-depth critique's judgment matches the ground truth for all steps. However, this filtering could introduce a selection bias, favoring problems where the teacher model is already proficient. The paper would benefit from an analysis of the characteristics of discarded samples.
*   **Data Generation for RL:** The automatic RL data construction using Monte Carlo sampling-based correctness estimation (from Wang et al., 2024) is a reasonable approach for scalability. The filtering criteria (discarding problems with all-correct or all-incorrect solutions) is pragmatic for creating a challenging dataset, but it potentially narrows the distribution of the training data. The authors acknowledge this trade-off but could discuss its potential impact on generalization to "easy" or "very hard" problems at test time.
*   **Logical Gaps:** The transition from SFT to RL is straightforward. However, the reward design is mentioned as a simple correctness reward (1 for correct final judgment, 0 otherwise). Appendix L explores an "informativeness-aware" reward but finds it leads to reward hacking (longer, unhelpful critiques). The choice of a sparse correctness reward is justified, but it raises a question: could a denser, step-level reward based on the consistency of the critique's reasoning further improve learning? The authors' ablation suggests not, but the exploration seems limited.
*   **Self-Improvement Setting:** The claim in Section 3.2.1 footnote and the experiment in Section 4.4 that the method works for self-improvement (using Qwen2.5-7B as its own teacher) is intriguing but not deeply explored. The performance gains are modest compared to using the 72B teacher, which is expected. This warrants more discussion on the limits and requirements for effective self-improvement within this pipeline.

**Experiments & Results:**
The experimental design is comprehensive, using three established benchmarks (MR-GSM8K, PRM800K, ProcessBench) and comparing against a wide array of baselines (PRMs, instruction-tuned LLMs, and advanced reasoning models like DeepSeek-R1).
*   **Supporting Claims:** The central claim—that DeepCritic outperforms existing LLM critics—is strongly supported. Table 1 shows DeepCritic-7B-RL-PRM800K achieves the best average F1 score among all critique models (69.1), beating GPT-4o (58.2) and same-sized DeepSeek-R1 models (63.4). The ablations (DirectDistill, InitialCritic) effectively demonstrate the importance of the separate step-wise and in-depth critique generation process.
*   **Baseline Fairness & Analysis:** The comparison is fair. The authors correctly note that Qwen2.5-Math-PRM-7B outperforms their model on error identification but lacks the ability to provide actionable feedback, which is a crucial distinction. They also astutely observe that DeepSeek-R1 models may be "directly solving" problems rather than critiquing, which could limit them on harder problems (lower scores on Omni-Math). This is a valuable insight.
*   **Test-Time Scaling & Refinement:** The test-time scaling experiments (Section 5) are a strength. Showing that majority voting (Maj@8) improves performance and that their model scales better than baselines is convincing evidence of the model's robustness. The critique-based refinement experiments (Table 3) demonstrate the practical utility of DeepCritic's detailed feedback. The improvement in generator accuracy (e.g., +3.4 points for Qwen2.5-7B on MATH500) is meaningful, though the authors should discuss statistical significance or provide confidence intervals, as the sample sizes (MATH500, AIME2024-25) are moderate.
*   **Weak-to-Strong Supervision:** The result that a 7B critic can help refine outputs of a 72B generator is compelling and aligns with recent interests in weak-to-strong generalization.
*   **Missing Analysis:** While the paper includes ablation on RL reward design and data construction noise (Appendices L & K), there is no analysis of *where* the model's improvements come from. An error analysis (e.g., does the model improve more on certain types of math errors, or on later reasoning steps?) would provide deeper insight into the model's learned capabilities.

**Writing & Clarity:**
The paper is generally well-written and logically organized. The methodology is clearly explained. Some minor points:
*   Figures and tables are referenced but not included in the provided text (e.g., Figure 1, 2, 5-7). This is presumably an artifact of the PDF extraction.
*   In Section 4.2, the sentence "We put the detailed results of separate accuracy..." has a formatting glitch ("Appendix ~~O~~").
*   The distinction between "critique models" and "process reward models (PRMs)" is maintained but could be emphasized earlier for readers less familiar with the terminology.

**Limitations & Broader Impact:**
The "Ethics Statement" is brief and focuses on the positive goal of enhancing oversight. A dedicated "Limitations" section is absent but critical for a complete paper.
*   **Key Limitations:** The work is heavily focused on **mathematical reasoning**. While Appendix M shows a promising extension to text summarization, this is preliminary. The generalizability to other complex, verifiable domains (e.g., code, science) and, more challengingly, to subjective domains is not fully established. The method's dependence on a reasonably capable teacher model for data curation (or a self-improvement loop) is a limitation for applying it to new domains or weaker base models. The computational cost of the data curation pipeline (multiple calls to a 72B model, Monte Carlo rollouts) is non-trivial.
*   **Broader Impact:** The paper briefly mentions the goal of "automated and scalable oversight." The societal impacts should be discussed more thoroughly. For instance, more capable critique models could be used to refine and strengthen harmful or misleading content, not just mathematical solutions. The potential for these models to be used in automated assessment systems also raises concerns about bias and fairness that are not addressed.

### Overall Assessment
This paper presents a solid and novel contribution. The proposed two-stage training pipeline for developing "deliberate" LLM critics is well-motivated, clearly described, and rigorously evaluated. The results demonstrate significant and meaningful improvements over strong baselines, including GPT-4o, on standard math critique benchmarks. The work also shows practical utility in helping generators refine outputs and exhibits promising test-time scaling properties. The main weaknesses are the lack of a formal limitations section, insufficient discussion of the societal implications, and a need for deeper analysis of error types and generalization beyond mathematics. For ICLR, these issues should be addressed in a revision, but the core technical contribution—a method to instill more thorough, self-critical reasoning into critique models—is significant and likely to meet the acceptance bar if these concerns are adequately mitigated.

# Neutral Reviewer
## Balanced Review

### Summary
This paper proposes DeepCritic, a two-stage training framework to enhance the critique capabilities of Large Language Models (LLMs) in mathematical reasoning. The first stage uses supervised fine-tuning (SFT) on a small, carefully curated dataset of "deliberate critiques," which include initial critiques followed by in-depth, multi-perspective verifications or meta-critiques of those initial critiques. The second stage employs reinforcement learning (RL) on either human-annotated data (PRM800K) or automatically generated data to further boost performance. The resulting 7B parameter model significantly outperforms existing LLM critics (including GPT-4o and specialized 7B models) on several error identification benchmarks and demonstrates effectiveness in helping generators refine erroneous solutions.

### Strengths
1. **Strong Empirical Results**: The proposed DeepCritic-7B-RL-PRM800K model achieves state-of-the-art performance on multiple math critique benchmarks (e.g., MR-GSM8K, PRM800K, ProcessBench), outperforming much larger models like GPT-4o and strong same-size baselines like DeepSeek-R1-Distill models. The performance gains over the base Qwen2.5-7B-Instruct (e.g., from 34.1 to 69.1 average F1) are substantial and well-documented (Table 1).
2. **Thoughtful Data Curation and Ablation Studies**: The process for generating the 4.5K SFT dataset—involving initial critique, in-depth critique (with multi-perspective verification and meta-critiquing), and final synthesis—is clearly described and justified. Ablation studies (comparing DirectDistill, InitialCritic, and DeepCritic SFT models in Table 1) effectively demonstrate that both the step-wise critique generation and the inclusion of in-depth critiques are crucial for the performance improvement.
3. **Analysis of Test-Time Scaling and Practical Utility**: The paper goes beyond simple accuracy metrics to show that the critic improves with majority voting (Appendix G) and, more importantly, that it can effectively assist LLM generators through verified majority voting and critique-based refinement, even showing weak-to-strong supervision potential (Table 3, Figure 4).

### Weaknesses
1. **Limited Controlled Comparison on Data Scale and Quality**: While the paper shows RL on 14.3K auto-generated data is effective, it does not fully disentangle the contribution of the novel SFT data format from the scale of the RL data. A stronger baseline would be an SFT-only model trained on a much larger dataset (e.g., the full PRM800K) to see if the "deliberate" format is the key factor rather than just more high-quality data.
2. **Superficial Exploration of RL Reward Design**: The RL stage uses a sparse accuracy reward. The paper mentions exploring an "informativeness-aware reward" but notes it did not help and led to reward hacking (Appendix L). This is presented as a negative result, but the investigation is minimal. A more thorough analysis of alternative reward functions (e.g., step-level correctness) could be valuable.
3. **Narrow Domain Focus and Preliminary Generalization Results**: The core contribution is evaluated exclusively on mathematical reasoning. While Appendix M presents a preliminary experiment on text summarization, it is limited in scope (200 samples, using GPT-4.1 as judge) and lacks the rigorous benchmarking and ablation studies performed for math. Claims of generalizability are therefore not fully substantiated.

### Novelty & Significance
**Novelty** is moderate. The idea of training a critic to generate longer, more deliberate Chain-of-Thought critiques is intuitive and builds directly on the literature about improving LLM reasoning and verification (e.g., PRMs, CriticCoT). The specific two-stage pipeline with iterative in-depth critique generation for SFT data curation is a clear and well-executed contribution. The use of Monte Carlo sampling for auto-labeling RL data follows existing work (Wang et al., 2024).

**Significance** is high for the ICLR community. Improving the quality and reliability of LLM self-critique is a central problem in scalable oversight and alignment. The paper provides a concrete, effective method that yields a powerful open-source critique model. The demonstrated test-time scaling properties and refinement utility make it practically relevant for improving reasoning systems.

### Suggestions for Improvement
1. **Conduct a more controlled ablation on data factors**. Train an SFT model on a larger dataset (e.g., all initial critiques from PRM800K) to isolate the impact of the *deliberate critique format* from the *data scale and teacher model strength*. This would strengthen the claim that the curation method is the key innovation.
2. **Deepen the analysis of the RL component**. Explore and discuss alternative reward formulations more thoroughly, perhaps incorporating dense, step-level rewards or investigating why the informativeness reward failed. Analyzing the training dynamics (e.g., how critique length and quality evolve during RL) would also be insightful.
3. **Strengthen the claims of generalizability**. To convincingly argue the framework applies to "subjective domains," expand the experiments in Appendix M. Use established summarization or dialogue evaluation benchmarks, compare against strong baseline critics, and perform similar ablation studies to show the necessity of the two-stage deliberate critique process in that domain.
4. **Improve the discussion of limitations and broader impact**. The paper briefly mentions computational constraints (max response length in RL). A more detailed discussion of the computational cost of the data curation and training pipeline would be useful for practitioners. Additionally, a brief discussion of potential misuse (e.g., generating more convincing but incorrect critiques) would be appropriate for an ICLR submission.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Ablation on the RL stage's contribution vs. simply using more SFT data.** The paper claims RL "further boosts" performance, but does not compare the RL stage to an SFT-only model trained on a comparable amount of additional high-quality critique data. Without this, the necessity and unique benefit of the RL stage is not established.
2. **Evaluation on out-of-distribution or more complex math datasets (e.g., Olympiad-level problems).** The benchmarks (GSM8K, MATH, PRM800K) are relatively standard. To claim the method develops "deliberate critiquing," it must be tested on problems where surface-level verification fails and deeper reasoning is required. Its performance drop on OlympiadBench (Table 1) suggests a fragility not adequately probed.
3. **Systematic comparison with strong, recent critique-specific baselines.** The paper compares against general LLMs and PRMs, but not against other state-of-the-art *critique models* trained for similar purposes (e.g., models from "Critique Fine-Tuning" (Wang et al., 2025) or "Teaching Language Models to Critique via RL" (Xie et al., 2025)). This omission makes it impossible to judge if the proposed pipeline is a meaningful advance over existing methods for training critics.
4. **Analysis of the reliability of the auto-labeled RL data.** The RL stage uses Monte Carlo sampling to label steps, but there is no validation of this method's accuracy on a held-out human-labeled set. If the auto-labels are noisy, the RL improvements might be due to dataset scale or other factors, not the proposed method's robustness.

### Deeper Analysis Needed (top 3-5 only)
1. **Quantitative analysis of what the model learns in each stage (SFT vs. RL).** Does SFT teach the *format* of deliberate critique, while RL simply sharpens *judgment accuracy*? An analysis correlating critique attributes (e.g., length, presence of specific phrases like "let me verify from another angle") with accuracy gains across stages is missing. This is needed to validate the core design.
2. **Analysis of failure modes and limitations.** The paper shows successes but does not analyze *when and why* DeepCritic fails. For instance, on which error types (conceptual vs. arithmetic) does it struggle? Does the critique sometimes become verbose but unhelpful? This analysis is critical for assessing the method's true utility and for future work.
3. **Breakdown of the "multi-perspective verification" and "meta-critiquing" behaviors.** The paper claims these are key innovations (Fig. 1), but only provides a coarse frequency statistic (Table 8) and a few hand-picked examples. A systematic analysis (e.g., using GPT-4 to categorize a large sample of generated critiques) is needed to prove these behaviors are consistently elicited and are not just artifacts of the SFT data format.

### Visualizations & Case Studies
1. **Side-by-side comparisons of critique quality for the same problem across baselines and different training stages.** Figure 1 shows one curated example. To convince readers of consistent improvement, the paper needs a figure showing several problems where Qwen2.5-7B-Instruct, DeepCritic-SFT, and DeepCritic-RL generate critiques, highlighting how depth and correctness evolve. This would directly visualize the claimed "deliberate" reasoning.
2. **Visualization of the refinement process guided by critiques.** Table 3 shows accuracy numbers after refinement. A case study figure tracing a wrong solution through the critic's feedback to the generator's corrected solution would demonstrate the practical value of the more "informative" feedback, which is a key claimed advantage over PRMs.

### Obvious Next Steps
1. **Test the framework on a non-mathematical, verifiable domain (e.g., code debugging or logical puzzles) within the main paper.** The claim of generality (footnote 1, Appendix M) is relegated to an appendix on a subjective task (summarization). Applying the *same* pipeline to a different structured reasoning domain (like code) in the main experiments would significantly strengthen the claim of a general method for training critics.
2. **Investigate the reward function design more thoroughly.** The paper dismisses an "informativeness-aware reward" in Appendix L because it led to reward hacking. However, a proper investigation into reward design (e.g., penalizing excessive length, rewarding conciseness alongside accuracy) is a logical next step that should have been explored more deeply, as the binary reward is a clear limitation.
3. **Analyze the cost-effectiveness trade-off.** The method produces long critiques. An analysis of whether similar judgment accuracy could be achieved with shorter, more focused reasoning (i.e., is all the verbosity necessary?) is missing. This is crucial for scalability, as long critiques increase inference cost for both the critic and the generator during refinement.

# Final Consolidated Review
## Summary
The paper proposes DeepCritic, a two-stage framework to enhance the critique ability of large language models in mathematical reasoning. The first stage uses supervised fine-tuning on a small, curated dataset of "deliberate critiques" that incorporate multi-perspective verification and meta-critiquing. The second stage applies reinforcement learning on either human-labeled or automatically generated data. The resulting 7B model outperforms existing LLM critics, including GPT-4o, on standard error identification benchmarks and demonstrates practical utility in helping generators refine solutions.

## Strengths
- The method achieves state-of-the-art performance on multiple math critique benchmarks (MR-GSM8K, PRM800K, ProcessBench), with the 7B model significantly outperforming larger models like GPT-4o and strong same-size baselines like DeepSeek-R1-Distill models.
- The data curation process—generating initial critiques followed by in-depth, multi-perspective verification and meta-critiquing—is well-designed and ablated to show its necessity over simpler distillation approaches.
- The framework demonstrates promising test-time scaling (e.g., improved accuracy with majority voting) and practical value by enabling effective critique-based refinement of generator outputs, including weak-to-strong supervision.

## Weaknesses
- The paper lacks a thorough analysis of limitations, such as the potential selection bias in SFT data filtering (where only samples with in-depth critiques matching ground truth are kept) and the impact of RL data filtering (discarding all-correct or all-incorrect solutions) on generalization to easier or harder problems.
- Claims of generalizability beyond mathematics are not strongly supported; the only evidence is a preliminary experiment on text summarization in an appendix, which lacks the rigor and ablation studies of the main math experiments.
- There is no validation of the reliability of the auto-labeled data used in RL, which is generated via Monte Carlo sampling. Without assessing label accuracy, it is unclear whether RL improvements stem from the method or from noisy supervision.
- Insufficient error analysis: the paper does not examine where the model fails, what types of errors it struggles with, or how the "deliberate" reasoning behaviors vary across stages, limiting insight into its capabilities and failure modes.

## Nice-to-Haves
- More controlled ablations to disentangle the contribution of the deliberate critique format from the scale of the RL data (e.g., comparing RL to SFT on a larger dataset).
- Deeper investigation into reward function design, though the paper already explores an informativeness-aware reward and reports negative results.
- Analysis of the cost-effectiveness trade-off due to the long critiques generated, such as whether similar accuracy could be achieved with shorter reasoning.

## Novel Insights
None beyond the paper's own contributions.

## Suggestions
- Add a limitations section discussing data biases, generalization boundaries, computational costs of the pipeline, and societal implications of more capable critique models.
- Conduct an error analysis to categorize failure modes (e.g., error types or step positions where the model errs) and quantitatively analyze the prevalence of multi-perspective and meta-critiquing behaviors in generated critiques.
- Validate the auto-labeling method on a held-out human-annotated subset to estimate label noise and ensure the robustness of RL improvements.

# Actual Human Scores
Individual reviewer scores: [6.0, 4.0, 4.0]
Average score: 4.7
Binary outcome: Reject
