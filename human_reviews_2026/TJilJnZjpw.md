# Read the Room: Video Social Reasoning with Mental-Physical Causal Chains

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
"Read the room", or the ability to infer others' mental states from subtle social cues, is a hallmark of human social intelligence, but remains a major challenge for current AI systems. Existing social reasoning datasets are limited in complexity, scale, and coverage of mental states, falling short of the rich causal dynamics found in real-life interactions. In this work, we introduce R$^3$-Bench, an evaluation benchmark with fine-grained annotations of belief, intent, desire, emotion, and their causal chains in complex scenarios. Furthermore, we introduce R$^3$-FDT, a large-scale training set generated through a novel automated pipeline with the same chain structure. We conduct a comprehensive evaluation of state-of-the-art (SOTA) large vision-language models (LVLMs) on R$^3$-Bench, revealing substantial deficiencies in consistent multi-step social reasoning. We also fine-tune a 7B model on R$^3$-FDT, achieving notable improvements across multiple relevant benchmarks. Our contributions are three-fold: (i) a novel benchmark with richly annotated, multi-step causal reasoning data; (ii) systematic evidence that SOTA LVLMs fall far short of human-level reasoning; (iii) a scalable training dataset that significantly enhances social reasoning performance. The datasets and code are available at: <https://github.com/LiXingNiu/Read-the-Room.git>.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors introduce a new dataset, $R^{3}$-VQA, to address the challenge of teaching AI to infer complex social and mental states from video. The dataset contains two parts: $R^{3}$-Bench, a high-quality video QA benchmark with fine-grained manual annotations for mental states (belief, intent, desire, emotion) and the multi-step causal chains that connect them to physical events; $R^{3}$-FDT, an training dataset augmented from existing datasets using a similar automatic framework. The authors showed that state-of-the-art multimodal LLMs fall short of human performance on $R^{3}$-Bench, and finetuning a model on $R^{3}$-FDT with GRPO yielded improvements across datasets.

### Strengths
- $R^{3}$-Bench contains rich, manual annotation of social reasoning chains. It also has decent scale (312 videos, 4,840 QAs) for an evaluation set 
- While $R^3$-FDT has no human-in-the-loop verification, using it for RL finetuning yielded significant improvements across datasets (including Social-IQ and IntentQA). The pipeline also seems scalable and adaptable.
- The difficult subset $R^{3}$-Bench-Hard showed significant gap between AI and human accuracy, demonstrating room for improvement.

### Weaknesses
- $R^3$-FDT is automatically generated, and the paper lacks crucial details on its verification. A "Hallucination Detection" step is mentioned, but the exact criteria for filtering, the specific prompts used, and the rejection rate (i.e., how many QA pairs were filtered out) are not reported. This makes it difficult to assess the final quality and potential noise level of the training data.
- The main evaluation set, $R^{3}$-Bench-DX, appears to be approaching saturation. The top-performing models, such as the finetuned Qwen2-VL-7B and Gemini 2.5 Pro, are close to human accuracy performance. This limits its long-term utility as a challenging benchmark.
- The "Hard" subset, which shows a much larger performance gap, is very small, containing only 316 questions. This limited scale and diversity make it difficult to rely on as a robust and comprehensive evaluation set for such a complex task.

### Questions
- Could the authors please elaborate on the construction of $R^{3}$-Bench-Hard? The paper states it was "sourced from the winning submissions of a social reasoning challenge", but what were the specific selection criteria? What attributes (e.g., causal chain length, type of mental state, presence of deception) make this subset quantifiably "harder" than $R^{3}$-Bench-DX? Given its small size, providing a few qualitative examples that directly compare a "Hard" question to a "DX" question would be beneficial.
- The novel consistency metric is appreciated, but its interpretation is difficult. The criterion -- where a chain is "consistent" only if all associated questions are answered correctly -- seems overly strict and may be the reason for the low consistency scores. The authors omit a human baseline for this metric, citing concerns about human memory. However, a baseline, even if imperfect, would be a good reference point to contextualize the models' low consistency scores. Could the authors provide this consistency score human baseline, or discuss why a modified human study was not feasible?
- The "high accuracy, low consistency" paradox  is ambiguous. The consistency metric doesn't distinguish between two very different issues: 1) A model answers a high-level causal question correctly without correctly answering the preceding, simpler questions in the chain. This would suggest the high-level question is flawed or hackable with a heuristic, not by causal reasoning. 2) A model answers basic event/mental state questions correctly but fails at the more complex, high-level reasoning questions that come later in the chain. This indicates a genuine model failure. Can the authors clarify whether the current metric distinguishes these two cases? A more granular analysis or metric seems necessary to determine if the low consistency is due to flawed questions (Case 1) or genuine model reasoning deficits (Case 2).
- Nit: there's an overlap of Table 1 on page 3

Overall, while I lean towards acceptance due to the dataset's demonstrated utility for fine-tuning, I have the above reservations about the saturation of the evaluation set and whether the benchmark robustly measure genuine social understanding.

### Soundness
3

### Presentation
3

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
This paper introduces R3-VQA, a benchmark (R3-Bench about 5K QAs on diverse videos) and training dataset (R3-FDT about 40k QAs on movie videos) for evaluating video-based social reasoning. The dataset annotates causal chains linking beliefs, intents, desires, and emotions to observable actions.  Evaluation of 10+ SOTA models shows that they achieve high accuracy on individual questions but fail to maintain consistency across related questions in the same causal chain. The authors propose chain consistency and subchain consistency metrics to measure this. Training Qwen2-VL-7B with GRPO on R3-FDT improves performance by +13% on R3-Bench-DX and +6% on external social reasoning datasets, suggesting that causal structure provides useful training signal for social reasoning.

### Strengths
* Systematic annotation of mental-physical causal chains in video with multiple mental state types; consistency metrics effectively expose gap  in SOTA models
* Chain consistency improves dramatically (5.48% → 19.63%) after training, demonstrating models learn coherent multi-step "social" reasoning 
* Rigorous construction: expert-verified annotations across multiple stages; comprehensive evaluation on 10+ models with accuracy and consistency metrics
* Strong competitive results: 7B model beats Gemini 1.5 Pro on hard set and approaches GPT-4o on standard benchmark; +6% generalization to external datasets (Social-IQ 2.0, IntentQA) also convincing
* Scalable automated pipeline (ARGUS) creatively leverages movie metadata to generate 41k training QAs; a valuable dataset for the community

### Weaknesses
* Related work somewhat underdeveloped: Dismisses MMToM-QA (2024, same year), Hi-ToM, SimpleTOM, and CausalChaos in single sentences without evidence or direct comparison—makes contribution difficult to situate relative to this work.
* Causal theory incomplete: Treats emotions as generic mental states rather than motivational drivers; ignores System 1/2 distinction between reactive (emotion-driven) and deliberative (belief-driven) reasoning. Also missing some ToM citations. Overall, the causal chains are reasonable but simplistic compared to cognitive processing.
* Human validation weak: No inter-rater agreement metrics. Consensus among annotators is not clearly discussed. It would be great if all the annotator information is provided in the dataset because social/affective labels are naturally ambiguous. Consistency in annotation is somewhat misguided.
* Domain characterization insufficient: Cross-domain gap acknowledged (movies → YouTube) but not quantified. No breakdown of R3-Bench by video type and no evaluation on genuinely real-world social scenarios.

### Questions
* MMToM-QA (2024) also annotates mental states in video. How does causal chain depth compare quantitatively? 
*  What is performance using standard SFT (not GRPO) on the same 13k training examples? 
* Can you provide error analysis showing whether model failures correlate with System 1 (reactive/emotion-driven) vs. System 2 (deliberative/belief-driven) scenarios? 
* Why are Cohen's kappa or Fleiss' kappa not reported for causal chain annotations? What was the disagreement rate between the two expert reviewers before consensus?
*  What % of R3-Bench videos are "real-life" vs. curated (ads/short films)? Does performance differ by video type? 
* Can you provide examples of Causal-Why questions where multiple answers seem valid? How were conflicts resolved?

### Soundness
2

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
3

### Summary
The paper introduces R3-VQA, a new video social-reasoning suite comprising (i) R3-Bench—an evaluation benchmark with fine-grained mental–physical causal chains and (ii) R3-FDT—a large-scale training set generated by an automated pipeline. The benchmark defines four QA types—Event Understanding, Mental State Estimation, Causal-Why, and Causal-How/What—generated from expert-annotated chains and verified by the domain experts. Reported scale: around 0.3k videos / 5k MC-QAs for R3-Bench and, for R3-FDT, about 3k videos / 68k MC-QAs (Table 1); elsewhere, the text states 2,812 videos / 41k QAs after quality filtering as well. Evaluations of many LVLMs demonstrate a stark gap between single-question accuracy and chain/subchain consistency (all QAs from a chain must be not wrong), highlighting fragmented reasoning. Fine-tuning Qwen2-VL-7B with GRPO on R3-FDT yields sizable gains on R3-Bench and transfers to Social-IQ 2.0 dataset / IntentQA dataset.

### Strengths
1. Explicit belief/intent/desire/emotion nodes and chain/subchain QA generation tie items to causal structure rather than one-off facts.
2. Two-stage expert checks for QAs and chains; a clear human-in-the-loop pipeline to keep the ambiguity low.
3. Chain and Subchain Consistency expose fragmented reasoning hidden by the average accuracy; formal definitions are offered.
4. Large consistency-accuracy gaps ( for example, GPT-4o 82.64% overall vs 25.36% chain consistency; Gemini 2.5 Pro 86.34% vs 36.60%) and dimension-wise weaknesses.
5. Automated R3-FDT pipeline (alignment, movie scripts, Gemini filtering, GPT-4o) plus GRPO improves Qwen2-VL-7B model ( for instance, plus 32.00% performance on R3-Bench-DX; and plus 9.81% on R3-Bench-Hard).

### Weaknesses
1. The OE scoring depends on an LLM judge. What is more, the sensitivity to prompts and judgment choice is not reported comprehensively. Frame or frame per second budgets differ comparatively over models. This confounds the absolute rankings to a considerable extent.
2. 3k/68k for R3-FDT is stated in Table 1; however, the text reports 2,812/41k later. It is better to clarify the final counts and which subset is utilized for the GRPO algorithm to make the paper clearer.
3. R3-FDT depends on movies and LLM-generated chains/QAs. The risk of stylistic bias or leakage still remains; in addition, the human IAA and leakage audits are not detailed.
4. Even though consistency is measured, wall-clock latency and user-utility vs delay are not evaluated directly.
5. The human chain or the subchain consistency is left out (with justification deferred to the Appendix), which limits context for the new metrics.

### Questions
1. How do OE and consistency scores change if the judge-LLM or prompt is varied? Any agreement with human raters on an OE slice?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces the video social reasoning task with fine-grained mental–physical causal chains. The paper releases an evaluation benchmark with verified Multiple choice-QA pairs built from annotated chains and a training set generated via a movie-script alignment and LLM pipeline. The paper introduces chain and subchain consistency metrics, evaluate many LVLMs and humans, and show GRPO fine-tuning of Qwen2-VL-7B improves accuracy and consistency.

### Strengths
- Task novelty: Causal-chain–grounded evaluation of belief, intent, desire, and emotion in video; chain and subchain consistency move beyond single-QA accuracy.
- Dataset annotation quality: Human-verified chains and QA, with explicit rules and hallucination filtering
- Significance: Reveals large accuracy-consistency gaps in SOTA LVLMs and provides training data that measurably improves a strong open model.
- Strong experiment results with ablation studies and generalization tests.

### Weaknesses
- Human study mismatch. Humans answer only one QA per chain, so their consistency is “severely underestimated,” complicating human–model comparisons. Remedy: redesign human protocol to mirror model setting or normalize metrics accordingly. 
- LLM-in-the-loop biases/leakage. GPT-4o is used to generate and self-correct chains and QAs; Gemini is used for hallucination detection. This can imprint model priors and evaluation artifacts. It'll be good to do ablation studies to assess its sensitivity to the design choice.
- Lack of uncertainty estimation. Many tables lack confidence intervals and seed variance.

### Questions
- how does the performance correlate with the chain length?

### Soundness
3

### Presentation
3

### Contribution
3
