# PTCG: Persona-guided Tree-based Counterargument Generation

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
The ability to generate counterarguments is important for fostering critical thinking, balanced discourse, and informed decision-making.
However, existing approaches typically produce only a single counterargument, thereby overlooking the diversity and persuasiveness required in real-world debates.
This limitation is critical, as the same topic may persuade different individuals only when framed from distinct perspectives.
To address this limitation, we propose Persona-guided Tree-based Counterargument Generation (PTCG), a framework that combines Tree-of-Thoughts–inspired step-wise generation and pruning with speaker persona selection.
By estimating the author’s persona from the original argument and incorporating speaker personas representing distinct perspectives, the framework operationalizes perspective-taking, enabling reasoning from multiple standpoints and supporting the generation of diverse counterarguments.
We propose a tree-based procedure that generates plans, selects the best, and produces multiple speaker persona-specific counterarguments, from which the most effective are chosen.
We evaluate PTCG through a comprehensive multi-faceted setup, combining Large Language Model (LLM)-as-a-Judge, classifier-based assessment, and human evaluations.
Our experimental results show that PTCG substantially improves both the diversity and persuasiveness of counterarguments compared to baselines.
These findings highlight the effectiveness of adaptive persona integration in boosting diversity and strengthening persuasiveness.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies the task of counterargument generation and introduces a persona-based approach with Tree-of-Thought (ToT) content planning. Specifically, given an original post (OP), the system first constructs three distinct personas, each representing a unique perspective. These personas then perform content planning via a Tree-of-Thought process before generating the final counterargument. All components are implemented through prompting a large language model (LLM). By decomposing the end-to-end generation and integrating persona creation, the proposed method aims to produce more diverse and audience-tailored arguments. Experiments conducted on the Reddit CMV dataset demonstrate the effectiveness of the approach with both automatic metrics and human evaluations.

### Strengths
- This paper addresses the important task of diverse counterargument generation, a challenging area where current LLMs still struggle to capture perspective diversity. The integration of persona-based modeling for argument generation is plausible.
- The work includes comprehensive automatic and human evaluations to effectively demonstrate the model’s performance.
- Overall, the paper is clearly written, logically structured, and easy to follow.

### Weaknesses
- My main concern is that the idea of using personas to enhance counterargument diversity appears very close to [1], which also creates personas and uses debate-style planning to broaden perspectives. The paper does not clearly articulate how it differs from [1], making the contribution feel less novel;
- Experiments rely primarily on relatively small base LLMs (<10B). It remains unclear whether the method provides gains when applied to stronger models (e.g., ChatGPT). A study on larger models would clarify the method’s robustness;
- Limited analysis of the paper: The paper lacks ablations (e.g., replacing ToT with CoT, change the number of personas) that would quantify each module’s contribution. It also omits error analysis and important experimental details such as inter-annotator agreement of human evaluation, sampling procedures, etc;
- Argument generation has a long history in NLP. The current related-work section is relatively brief; a more comprehensive synthesis—covering classical approaches, persona-based generation, planning methods (CoT/ToT), and debate frameworks—would strengthen the paper’s positioning.

[1] Debate-to-Write: A Persona-Driven Multi-Agent Framework for Diverse Argument Generation, COLING 2025



*Note: At this stage, I lean toward a rejection rating, primarily due to my concern regarding Point 1 (novelty overlap). However, I would be open to revising my score depending on how convincingly the authors address this issue in their rebuttal.*

### Questions
See Weaknesses

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a generation of counterarguments with the Tree-of-Thought-inspired method called Persona-guided Tree-based Counterargument Generation (PTCG). The authors cluster different persona types and first, define the personality cluster of the claim author, then find personas from the nearest and the farthest cluster, and then generate counterarguments based on the perspectives of the selected personas. They use 847 threads from ChangeMyView dataset. The evaluation stage includes automatic metrics, classifier-based and LLM-as-a-Judge, and human evaluation. The method is compared with other baselines and shows that PTCG improves diversity and persuasiveness.

### Strengths
The authors provided a clear problem definition in counterargument generation and persuasiveness and propose a rigorous pipeline to generate counterarguments from different perspectives. Moreover, they provide a comprehensive evaluation of the validity of their proposed method by reporting Human and Automatic Evaluation results by comparing their method with the baseline solutions.

### Weaknesses
- **Lack of comprehensive qualitative analysis.** There are only surface-level explanations in the paper. Perhaps authors could have enhanced a little more on how exactly the counterargument generated by PTCG differed from the baseline, i.e., the length of the generated text, what types of points/perspectives where the most persuasive, etc.
- No agreement in human evaluation reported.

### Questions
- How does each persona (nearest, farthest) affect the argumentation style?
- address the points listed in the weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes PTCG (Persona-Guided Tree-Based Counterargument Generation) — a framework that integrates persona-based conditioning with Tree-of-Thoughts (ToT)-inspired step-wise generation and pruning to produce multiple, diverse, and persuasive counterarguments.
The key idea is to estimate the persona of the original argument’s author (the “OP persona”), select three personas from distinct clusters (same, nearest, furthest), and use a structured reasoning tree to plan and generate counterarguments from those perspectives. The method is evaluated on 847 posts from the ChangeMyView (CMV) subreddit, comparing against multiple baselines using both LLM-as-a-Judge and human evaluations.
Results indicate improvements in persuasiveness, diversity, and stance quality compared to strong LLM baselines (e.g., Llama-3.1-8B, DeepSeek-R1).

### Strengths
1. **Novel Integration of Personas with Tree-of-Thoughts Reasoning**  
   - The framework creatively combines persona conditioning and step-wise reasoning for counterargument generation.  
   - The method’s design (Figure 2, p. 3) is well thought out — the combination of OP persona estimation, persona selection across cluster distances, and tree-based reasoning is intuitive and well motivated by psychological theory on perspective-taking.

2. **Clear Empirical Gains**  
   - Quantitative improvements are consistent across metrics (Table 1, p. 6; Table 2, p. 7).  
   - Especially notable is the improvement in both general and targeted persuasiveness (8.26 vs 8.07 LLM baseline) and diversity (4.27 vs 4.21).  
   - Human and LLM-based pairwise evaluations (Figure 3, p. 8) corroborate these gains robustly.

3. **Comprehensive Evaluation Setup**  
   - Combines LLM-as-a-Judge, classifier-based, and human evaluation.  
   - Includes ablations isolating tree-based and persona modules (Table A4, p. 18).  
   - Tests robustness across multiple evaluators (Appendix A4, p. 16) using DeepSeek and Qwen.  
   - Detailed prompts for reproducibility (Appendix A3–A11).

4. **Strong Theoretical Grounding**  
   - Connects the system design to psychological theories of perspective-taking and empathy (Batson 1997; Green & Brock 2000).  
   - Demonstrates understanding of argumentation literature and connects to recent LLM works on personalized persuasion.

5. **High Presentation Quality**  
   - Writing is clear, figures are illustrative (Figure 1 and 2 visually explain contributions well), and tables are well formatted.  
   - Ethical considerations and reproducibility statements are thorough (p. 10).

### Weaknesses
**Insufficient Novelty Beyond Composition**  
   While the combination of persona-guided reasoning and ToT is elegant, each component individually is drawn from existing techniques:  
   - Persona grounding via clustering and embedding similarity (e.g., PersonaHub 2024)  
   - Tree-of-Thoughts reasoning (Yao 2023)  
   The paper’s novelty thus lies mainly in their integration for counterargument generation, which may be seen as an engineering composition rather than a fundamentally new algorithmic contribution.  
   **Suggestion:** Strengthen the theoretical justification — e.g., show why persona distance correlates with persuasive diversity, or include a formal analysis of ToT pruning effects on diversity/persuasiveness trade-offs.

2. **Evaluation Metrics Depend Heavily on LLM-as-a-Judge**  
   - Over-reliance on GPT-based evaluators (including GPT-4o-mini) introduces potential bias.  
   - Even though multiple evaluators are used, there is no clear calibration or validation of LLM judgments against human ground truth beyond the small human study (n = 5, 50 samples).  
   **Suggestion:** Increase human evaluation coverage or report inter-annotator agreement and effect sizes to support claims.

3. **Limited Dataset Scale and Generalization**  
   - The evaluation dataset (847 posts) is relatively small and domain-specific (ChangeMyView).  
   - It is unclear whether the model generalizes to other argumentation domains (e.g., legal reasoning, political debates, or social media outside CMV).  
   **Suggestion:** Test cross-domain transfer or include few-shot qualitative examples from unseen domains.

4. **Lack of Ablation on Persona Clustering Design**  
   - Persona clustering (HDBSCAN → 39 clusters) is a critical design choice, but Table A2 only shows clustering metrics (Silhouette 0.65).  
   - There is no analysis of how the number of clusters or persona distance selection impacts results.  
   **Suggestion:** Add sensitivity experiments varying cluster counts (e.g., 20 vs 50 clusters) or persona selection strategies (e.g., random vs distance-based).

5. **Interpretability and Example Depth**  
   - While Table A1 provides one case study, qualitative discussion is limited to general claims.  
   - No explicit link between specific persona archetypes and rhetorical styles.  
   **Suggestion:** Provide detailed case studies showing how persona identity (e.g., “developer,” “artist”) concretely changes argumentative framing.

6. **Clarity on Computational Cost and Efficiency**  
   - The multi-step generation (3 personas × 3 plans × 3 candidates) implies ~27 LLM calls per input.  
   - There is no mention of inference time or cost efficiency compared to simple sampling.  
   **Suggestion:** Add runtime analysis and scalability discussion — essential for practical deployment or large-scale debates.

### Questions
1. How does persona distance quantitatively relate to persuasion diversity or success?  
2. Were any sanity checks performed to ensure persona clusters are semantically meaningful (vs. random partitions)?  
3. How sensitive are the results to the choice of backbone LLM (e.g., GPT-4 vs. Qwen-2.5 vs. Claude)?  
4. What is the average runtime or token cost per example during inference?  
5. Would the framework scale to real-time debate settings with human participants?  
6. Can the same persona–ToT integration framework generalize to tasks beyond persuasion, such as negotiation or empathy modeling?

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
2

### Summary
This paper proposes PTCG (Persona-Guided Tree-based Counterargument Generation) , a framework that integrates persona grounding and Tree-of-Thoughts (ToT) style reasoning to generate diverse and persuasive counterarguments. The system estimates the persona of the original author (OP) from an input argument, selects three contrasting speaker personas (same, nearest, and furthest cluster), and then employs a tree-based generation process to plan, prune, and produce multiple counterarguments from distinct perspectives.

### Strengths
Novel Integration of Persona and Structured Reasoning: The combination of persona-guided conditioning and Tree-of-Thoughts reasoning is original and well-motivated. The design systematically operationalizes “perspective-taking” an idea rooted in social psychology for counterargument generation

### Weaknesses
1. Limited Scalability and Efficiency Discussion: The tree-based process requires multiple generations and evaluations per persona, but the paper lacks analysis of computational cost or inference latency.

2. Heavy reliance on LLM-as-a-Judge (GPT-4o-mini) may inflate improvements, as PTCG uses similar models during generation. Although human evaluation is included, the sample (50 cases × 5 raters) is relatively small.

### Questions
Please refer to the weakness part.

### Soundness
2

### Presentation
3

### Contribution
2
