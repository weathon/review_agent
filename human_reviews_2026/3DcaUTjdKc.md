# Paper2Code: Automating Code Generation from Scientific Papers in Machine Learning

- Decision: Accept (Poster)
- Scores: 8, 6, 8, 0

## Abstract
Despite the rapid growth of machine learning research, corresponding code implementations are often unavailable, making it slow and labor-intensive for researchers to reproduce results and build upon prior work. In the meantime, recent Large Language Models (LLMs) excel at understanding scientific documents and generating high-quality code. Inspired by this, we introduce PaperCoder, a multi-agent LLM framework that transforms machine learning papers into operational code repositories. PaperCoder operates in three stages: planning, where it constructs a high-level roadmap, designs the system architecture with diagrams, identifies file dependencies, and generates configuration files; analysis, which focuses on interpreting implementation-specific details; and generation, where modular, dependency-aware code is produced. Moreover, each phase is instantiated through a set of specialized agents designed to collaborate effectively across the pipeline. We then evaluate PaperCoder on generating code implementations from machine learning papers based on both model-based and human evaluations, particularly from the authors of those papers, with author-released repositories as ground truth if available. Our results demonstrate the effectiveness of PaperCoder in creating high-quality, faithful implementations. Furthermore, it consistently shows strengths in the recently released PaperBench benchmark, surpassing strong baselines by substantial margins. Code is available at: https://github.com/going-doer/Paper2Code.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces PaperCoder, a multi-agent large language model (LLM) framework that automatically converts machine learning research papers into functional, repository-level code implementations. The system operates in three structured phases: planning, analysis, and coding. PaperCoder is evaluated on a benchmark of recent ML papers (ICLR, ICML, NeurIPS) and compared against baselines like ChatDev and MetaGPT, using both model-based and human evaluations. Results show that PaperCoder consistently generates more accurate and executable implementations, outperforming all baselines on both fidelity and usability metrics.

### Strengths
1. The paper tackles a highly impactful and underexplored task—automatic code generation from scientific papers—which directly addresses the reproducibility crisis in ML research. The framing of “Paper2Code” as a structured, multi-agent reasoning pipeline is both original and well-motivated.

2. The decomposition into Planning–Analysis–Coding is conceptually sound and empirically justified. Each stage mirrors human reasoning during software development, which gives the approach interpretability and modularity. The inclusion of architecture diagrams, execution order reasoning, and configuration synthesis adds practical depth.

3. The paper conducts extensive quantitative and qualitative evaluations, showing the effectiveness.

### Weaknesses
1. While the use of LLMs as judges is increasingly common, it introduces potential circularity and bias, especially since the same family of models may be used for both generation and evaluation. More independent human-based verification (beyond author feedback) would strengthen the claims.

2. Although executability is discussed, actual reproduction of original experimental results remains limited (only partial success in 4/5 cases). The paper does not sufficiently quantify semantic correctness or result fidelity, which are crucial for scientific reproducibility.

3. The ablation study (Table 6) shows some counterintuitive drops (e.g., architecture design hurting performance). While the authors provide a plausible rationale, a deeper qualitative analysis of failure cases would improve interpretability.

### Questions
none

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces PaperCoder, a multi-agent large language model (LLM) framework designed to transform machine learning papers into functional code repositories. The system emulates a human development pipeline, consisting of three key stages: planning, analysis, and generation. The authors evaluate PaperCoder using two benchmarks: Paper2CodeBench and the newly released PaperBench, employing both automated reference-free and reference-based metrics, along with expert human assessments. The results demonstrate significant improvements over baselines: 88% of the generated repositories receive the highest ratings, 92% of human judges find them useful, and the generated code typically requires only minor adjustments (with an average of 0.81% of lines modified). Ablation studies confirm the contribution of each stage to the overall performance. In summary, this paper addresses a valuable problem, and the proposed methodology and evaluation approach will provide meaningful support for future research.

### Strengths
1.	This paper introduces PaperCoder, a multi-agent large language model (LLM) framework designed to transform machine learning papers into functional code repositories through three key stages: planning, analysis, and generation. The results demonstrate significant improvements over baselines and ablation studies in Table 6 demonstrate the effectiveness of each proposed component.
2.	This paper constructs Paper2CodeBench by collecting recent machine learning papers with available code, filtering repositories for manageable size, and selecting high-quality papers through model-based evaluation. The benchmark provides reliable evaluation support for future research on paper-to-code generation.

### Weaknesses
1.	Reproducibility is a crucial metric for assessing code quality. However, this study conducts reproducibility analysis on only a limited subset, while the primary evaluation relies on the LLM’s judgment of code quality. It remains unclear whether the LLM’s scores reliably correlate with actual reproducibility, i.e., whether code receiving higher scores is truly more reproducible.
2.	The selected papers are drawn from ICLR, ICML, and NeurIPS 2024, along with their associated code repositories. It is important to consider potential data leakage risks, as these codebases may have been incorporated into the training of open-source models or closed-source APIs.
3.	In the LIMITATIONS section, the authors note that the current model only processes textual inputs. However, many methods and model architectures depend on the joint interpretation of diagrams and text. Consequently, relying solely on textual information may not faithfully reproduce the intended model structures. Exploring the use of PDFs as images input to a multimodal LLM for code generation represents an interesting direction.

### Questions
See the weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
*Summary*

This paper addresses the critical reproducibility challenge in machine learning, where a significant fraction of published papers (e.g., ~80.5% in 2024) do not release their code. To combat this, the authors introduce **PaperCoder**, a multi-agent Large Language Model (LLM) framework designed to automatically generate functional code repositories directly from scientific papers

The core of PaperCoder is a structured, three-stage pipeline that mimics a human developer's workflow:
1.  **Planning:** A set of specialized agents first constructs a high-level roadmap, designs the system architecture (including class and sequence diagrams), identifies file dependencies and execution order, and generates configuration files (e.g., `config.yaml`)
2. **Analysis:** The framework then performs a fine-grained interpretation of the plan, defining implementation-specific details, inputs/outputs, and algorithmic constraints for each module
3.  **Generation:** Finally, the framework synthesizes the modular, dependency-aware code file-by-file, using all previously generated artifacts as context.

To evaluate their framework, the authors introduce a new benchmark, **Paper2CodeBench**, consisting of 90 recent papers from ICLR, ICML, and NeurIPS 2024. They conduct extensive evaluations using both model-based (reference-free and reference-based) and expert human evaluations, notably involving the authors of the original papers. The results demonstrate that PaperCoder substantially outperforms strong baselines, with 92% of human judges reporting the generated repositories as helpful.

### Strengths
This paper makes a highly significant contribution to the ICLR community.

1 **Significance of Problem:** The work directly addresses the critical and widely-felt reproducibility crisis in machine learning, where the lack of code is a major barrier to scientific progress.

2 **Novel and Sound Methodology:** The proposed **PaperCoder** framework is a strong contribution. Its multi-stage (Planning, Analysis, Coding) pipeline is a logical and novel approach that mimics how a human expert would translate a paper to code. The "Planning" stage, with its decomposition into an overall plan, architectural design, logic design, and configuration generation, is particularly strong and provides the necessary scaffolding that simpler one-shot methods lack.

3 **Exceptional Evaluation Protocol:** The evaluation is a major strength.
    1.  The authors introduce **Paper2CodeBench**, a new, relevant, and large-scale benchmark of 90 papers from top 2024 conferences.
    2.  They employ a robust LLM-as-judge evaluation with both reference-based (vs. author code) and reference-free (vs. paper text) settings.
    3.  Most impressively, they validate their findings using **human evaluations from the original paper authors**, which is the highest possible standard for this task. The strong results from this (e.g., 92% of authors found the generated code helpful) are highly persuasive.

4 **Clarity and Presentation:** The paper is extremely clear, well-written, and easy to follow. The figures and tables are informative and effectively communicate the framework's design and results.

### Weaknesses
1 **Overstated Executability Claims:** The abstract's claim of "functional code" and the conclusion's "strong executability" are misleading. These claims appear to rest on a *manual* debugging analysis of only **five** papers, which found an 0.81% fix rate. This is a very small and likely biased sample.

2 **Contradictory Automated Execution Results:** A more comprehensive analysis in Appendix B.4 (Table 12) shows that when the generated code was run automatically on the *full 90-paper benchmark*, only **4 out of 90** repositories executed successfully. This suggests the system is not "near-executable" out-of-the-box.

3 **Failure to Address Environment Setup:** The primary failure mode identified in Table 12 is environmental: "Missing Dependency" (23 papers), "ImportError" (14 papers), and "ModuleNotFoundError" (14 papers). Although the "Logic Design" phase plans for "Required packages", the generation stage clearly fails to implement this reliably. The framework is good at generating Python *syntax* but poor at generating the *runnable environment* (e.g., a `requirements.txt` file).

4 **Unresolved Ambiguity Bottleneck:** The paper's own fine-grained analysis (Figure 5) reveals that the framework's implementation coverage is lowest for **"Data Processing" (56%)**. The authors attribute this to papers being "under-specified", but handling this ambiguity *is* the core difficulty of the task. A case study (Table 15) confirms this, showing one failure was due to an "overly simplified" loss function description. The framework currently guesses (and fails) in the face of ambiguity.

5 **Potential Evaluation Bias:** In the human evaluation (Table 14), the top reasons for preferring PaperCoder were "Completeness" and "Clean Structure". Since the PaperCoder framework is *explicitly designed* to produce a complete, modular structure (via its Planning stage), the evaluation may be rewarding the framework's *scaffolding* more than its *algorithmic correctness* when compared to one-shot baselines that lack this scaffolding by default.

### Questions
1.  Given that only 4/90 automated runs succeeded and the primary failure was dependency-related, why isn't a "DevOps Agent" (to generate `requirements.txt`, `Dockerfile`, etc.) a core part of the "Planning" and "Generation" stages? This seems like a crucial missing piece for generating "functional repositories."
2.  The paper uses "LLM-assisted debugging" as an *evaluation* step in Section 4.3. This proved effective. Have the authors considered integrating this as a *closed-loop refinement step* within the PaperCoder framework itself? For example, the framework could generate the code, attempt to execute it in a sandbox, catch the `stderr`, and feed the error message back to the "Coding" agent to self-correct. Or, as an alternative, leverage unit tests for all the components.
3.  The "Data Processing" stage has the lowest coverage (56%), and a case study failed on an ambiguous loss function. How does the framework currently handle ambiguity? Would it be feasible for the "Analysis" agent to *stop and query the user* when it detects high ambiguity (e.g., an "overly simplified" description), rather than proceeding to generate a potentially incorrect implementation? Or, at least marking potential issues coming from unders-specification of the paper's description in the code?

### Soundness
3

### Presentation
4

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
The paper presents paper2code, an ambitious system for translating scientific ML papers into executable code. The pipeline comprises three stages—planning, analysis, and coding—aimed at extracting algorithmic intent and producing runnable implementations. Evaluations on the authors’ selected metrics indicate promising effectiveness of the approach.

### Strengths
1. Framing end-to-end paper to code generation is both challenging and potentially transformative for reproducibility and rapid prototyping.
2. The decomposition into planning, analysis, and coding is reasonable and provides a clear, modular workflow for bridging natural language descriptions and implementation.

### Weaknesses
1. It is not evident how the paper defines “correct” generated code (e.g., functional equivalence, numerical agreement within tolerance, adherence to algorithmic specifications). Please clarify ground truth, acceptance tests, and evaluation thresholds.
2. The claim that fixes average 0.81% of total lines may understate practical burden: small line edits can involve significant debugging time. Consider reporting developer time, number of repair iterations, error categories (syntax/import/runtime/logic), and success rates after each repair round.
3. Correct coding implementations can vary (e.g., refactoring choices, hyperparameters, library calls). Explain how the evaluation accommodates non-uniqueness—e.g., via canonical reference implementations, specification-driven unit tests, layout-/refactor-invariant metrics, or behavior-based measures on hidden test sets.

### Questions
1, How is correctness evaluated when multiple valid implementations exist, and what does the reported 0.81% edit rate imply for developer time and repair iterations?
2. How robust is the system across domains/frameworks/formats, how are licensing/leakage and security risks handled, and where does human review add the most value?

### Soundness
1

### Presentation
1

### Contribution
2
