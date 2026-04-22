# PuzzleWorld: A Benchmark for Multimodal, Open-Ended Reasoning in Puzzlehunts

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 8, 6, 2

## Abstract
Puzzlehunts are a genre of complex, multi-step puzzles lacking well-defined problem definitions. In contrast to conventional reasoning benchmarks consisting of tasks with clear instructions and constrained environments, puzzlehunts requires discovering the underlying problem structure from multimodal evidence and iterative reasoning, mirroring real-world domains such as scientific discovery, exploratory data analysis, or investigative problem-solving. Despite progress in foundation models, their performance on open-ended settings remains largely untested. We introduce PuzzleWorld, a comprehensive benchmark of 667 puzzlehunt-style problems designed to assess step-by-step, open-ended, and creative multimodal reasoning. Each puzzle is annotated with the final solution, detailed reasoning traces, and cognitive skill labels, enabling holistic benchmarking and fine-grained diagnostic analysis. Most state-of-the-art models achieve only 1-4\% final answer accuracy. On PuzzleWorld, the best model solves only 18\% of puzzles and reaches 40\% stepwise accuracy, matching human puzzle novices but falling significantly behind puzzle enthusiasts. To demonstrate the value of our reasoning annotations, we show that fine-tuning a small model on reasoning traces boosts stepwise accuracy from 4\% to 11\%, which translates to improvements in downstream visual reasoning tasks. Our detailed error analysis reveals that current models exhibit myopic reasoning, are bottlenecked by the limitations of language-based inference, and lack sketching capabilities crucial for visual and spatial reasoning. We release PuzzleWorld at https://github.com/MIT-MI/PuzzleWorld to support future work on building more general, open-ended, and creative reasoning systems.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
PUZZLEWORLD introduces a benchmark of 667 puzzlehunt-style problems designed to evaluate multimodal, open-ended reasoning in AI systems. Unlike conventional benchmarks with well-defined tasks, puzzlehunts require discovering problem structures from ambiguous, multimodal clues, mirroring real-world challenges like scientific discovery or investigative problem-solving. The dataset includes detailed annotations of solutions, step-by-step reasoning traces, and cognitive skill labels. Evaluations show state-of-the-art models achieve only 1–14% final answer accuracy, with the best model (GPT-o3) matching novice human performance but lagging behind enthusiasts and experts. The benchmark also enables diagnostic analysis, revealing model limitations in backtracking, visual reasoning, and sketching capabilities. Fine-tuning on reasoning traces improves stepwise accuracy, demonstrating the utility of PUZZLEWORLD for advancing general-purpose reasoning.

### Strengths
1. PUZZLEWORLD fills a critical gap by focusing on open-ended, discovery-driven problems rather than constrained tasks. Its emphasis on multimodal clues and unstructured problem-solving aligns with real-world scenarios, providing a more holistic evaluation of reasoning capabilities.
2. The dataset includes meticulously curated step-by-step reasoning traces, cognitive skill labels, and modality tags. These annotations support fine-grained diagnostics, error analysis, and model training, surpassing existing benchmarks in depth and utility.
3. The study comprehensively tests frontier models (e.g., GPT-o3, Claude Opus) and human baselines, highlighting significant performance gaps. Stepwise accuracy metrics offer nuanced insights beyond final answers, revealing intermediate reasoning failures.

### Weaknesses
1. I think the experimental analyses are not sufficient. For example, we see that almost all MLLMs perform poorly, but the reasons behind this are worth further investigation. Is the model's poor performance due to poor performance in the visual modality or poor performance in the text modality? In my experience, most MLLMs are unable to perform well in downstream tasks due to their insufficient ability to recognize text in images.
2. The article lacks discussion of more related work, such as [1].
3. I look forward to seeing more case studies.

[1] LatEval: An Interactive LLMs Evaluation Benchmark with Incomplete Information from Lateral Thinking Puzzles.

### Questions
See the above weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors introduce a dataset of 667 "puzzle hunt"-style puzzles, complete with detailed annotations of reasoning for how to properly solve the puzzle. The authors discuss how this dataset is constructed, and how the annotations are carefully done to ensure no ambiguity in puzzle answers. They evaluate frontier models on the puzzles, and report the performance of both closed- and open-source models. The paper then investigates the effect of fine-tuning on reasoning traces on better performance on other puzzles, and expounds a detailed error analysis of current-generation VLMs.

### Strengths
I really like this dataset, both in terms of the intrinsic open-endedness of the puzzles, and the fact that the puzzles have an unambiguous and easy-to-evaluate final answer; I think this is a strong contribution to the field. I also appreciate the attention paid to making the dataset correct and unambiguous. The paper itself is well-written, and the figures are informative. I like the detailed analysis in section 5.4, and appreciate the contamination check in C.1. In all, this is a valuable dataset which was constructed carefully, and an excellent paper analyzing current performance.

### Weaknesses
No obvious weaknesses beyond the classic advice that more data points would be better :) Also, it would be nice to re-evaluate on all the new models that came out since the submission deadline, if that's not too hard!

### Questions
Would the MIT Mystery Hunt archives be a good source to expand a dataset like this?

### Soundness
4

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
This paper presents PUZZLEWORLD, a benchmark of 667 real-world puzzlehunt problems for evaluating multimodal, open-ended, multi-step reasoning. Each puzzle includes detailed reasoning traces, modality and cognitive-skill labels, enabling fine-grained diagnostic analysis. Experiments show that even frontier models achieve only 1–14% final accuracy, while fine-tuning on reasoning traces improves stepwise accuracy from 4.8% to 11%. The benchmark is clear, rigorous, and highly relevant to general reasoning research.

### Strengths
(1) Proposes the first large-scale open-ended puzzlehunt benchmark, transforming real Puzzled Pint puzzles into machine-readable tasks that test discovery-driven reasoning.

(2) A two-stage GPT-4o + human verification pipeline ensures 96.5% correctness and no contamination.

(3) Evaluates eight major models with both final and stepwise metrics, identifying systematic weaknesses (myopic reasoning, language bottleneck, lack of sketching).

(4) Reasoning-trace supervision improves intermediate reasoning and transfers to Rebus and MathVista tasks.

### Weaknesses
(1) The paper omits discussion of FINEREASON (Chen et al., 2025; arXiv:2502.20238), which also studies reflective puzzle reasoning through step decomposition. Comparing PUZZLEWORLD’s open-ended, multimodal puzzles with FINEREASON’s structured logic ones would clarify its unique scope.

(2) Limited ablations on annotation noise and prompting baselines; fine-tuning details appear only in the appendix.

(3) Dataset excludes audio/video modalities and depends on OCR, limiting breadth.

### Questions
– Please add a short discussion contrasting PUZZLEWORLD with FINEREASON, emphasizing that PUZZLEWORLD extends structured reflective reasoning to multimodal, discovery-driven contexts.

– Evaluate robustness to imperfect annotations and potential cross-benchmark transfer.

– Future work could merge PUZZLEWORLD’s multimodal puzzles with reflective reasoning frameworks for richer diagnostics.

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
4

### Summary
The paper introduces PUZZLEWORLD, a benchmark compiles 667 real puzzlehunt problems to test open-ended multimodal reasoning, adding stepwise solution traces, modality and skill tags, and an LLM-judge for intermediate scoring; frontier models perform poorly overall (best ≈14% answer accuracy, ≈40% stepwise), fine-tuning on reasoning steps improves stepwise scores but not final accuracy.

### Strengths
- open-ended puzzles obtained from real world, filling a current research gap. 

- the paper demonstrated a clear performance gap that stresses present models; best model only has ~14% answer accuracy

### Weaknesses
- the work is very interesting, but unfortunately not very thoroughly done, for examples:
    - The “easy/medium/hard” tags come from original puzzle metadata rather than a benchmark-defined rubric. There is no formal difficulty calibration, no solver-time distribution analysis, and no additional validation along the annotation. 

   - Humans are grouped into novice/enthusiast/expert tiers, but there is no modality-level breakdown and no step-level scoring symmetry with models. In addition, the human evaluation is only for 5% of the samples, thus the comparison in table 2 is not as informative as it appears. 

- The benchmark comes with one gold standard chain per puzzle , yet there is a chance the puzzlehunts can have multiple legitimate solve paths. There is no mechanism for crediting alternative correct partial chains or heuristic leaps, thus the evaluation of the reasoning paths can be quite limited.

- the significance of the benchmark may be limited in the sense that the problems and solutions are quite scoped with the culture of puzzlehunt. These specialized cognitive challenges might not well generalize to a broader scope.

### Questions
I will recommend the authors to increase the rigor of the benchmark, as discussed above.

### Soundness
2

### Presentation
3

### Contribution
2
