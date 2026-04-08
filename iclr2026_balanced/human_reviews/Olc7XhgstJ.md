## Human Reviewer 1

### Summary
The paper proposes a novel framework called Steady Thought, which aims to mitigate the pervasive phenomenon of "under-thinking" in Large Reasoning Models during complex reasoning tasks. This phenomenon is characterized by the model's failure to persevere and fully explore a promising reasoning path, instead switching excessively and inefficiently between thought trajectories.

### Strengths
1.The writing is clear, and the motivation is well articulated.

2.Recognizing the sensitivity of DPO to length bias, the authors introduce a length-normalized STPO objective based on SimPO, which is crucial for their method since the rejected switching trajectories are typically much longer than the selected completions.

3.Both accuracy and efficiency are improved.

### Weaknesses
1.The paper states that STPO reduces the number of tokens. For very challenging problems such as AIME 2024, the model may need multiple switches to find the correct reasoning path, indicating that exploration is valuable. Does ST risk over-penalizing reasonable exploration and switching, and to what extent might this affect the model’s ability to initially explore diverse reasoning strategies?

2.The core preprocessing step, thought segmentation, relies on entropy-based detection and predefined thresholds. Although the authors mention hyperparameter tuning, there is a lack of analysis on the stability and robustness of these thresholds across different model scales (1.5B vs. 8B) and task types (mathematics vs. programming).

3.In the thought completion stage, the model prevents switching by directly lowering the logits of trigger words such as “wait” and “alternatively.” This heuristic intervention might contradict the goal of STPO, which aims to implicitly suppress such words through learning.

### Questions
see weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
4

---

## Human Reviewer 2

### Summary
This paper, “SteadyThought: Mitigating LLM Under-Thinking via Thought-Level Preference Optimization,” addresses the phenomenon of under-thinking in large reasoning models (LRMs)—a tendency to switch reasoning trajectories excessively, abandoning promising thoughts prematurely. To solve this, they propose Steady Thought (ST), which consists of thought segmentation, thought completion, and fine-grained preference optimization. Experiments on multiple reasoning models across datasets demonstrate the effectiveness of the proposed approach.

### Strengths
(1) This paper addresses the frequent thought switching problem by Steady Thought, a thought-level preference optimization framework.

(2) Thought segmentation and thought completion are used to construct preference pairs to optimize LLMs.

(3) Experiments are tested on two large reasoning models across three datasets.

### Weaknesses
(1) The reasonability of using entropy to segment thoughts is not well justified. 

(2) The technical depth and novelty of the proposed method is somewhat limited.

(3) The results in Table 2 are questionable. The percentage of correct thoughts is reduced when using steady thought. To my understanding, steady thought should reduce the number of thoughts but increase the percentage of correct thoughts.

### Questions
Is there any quantitative metric to show the effectiveness of using entropy to segment thoughts?

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
4

### Confidence
3

---

## Human Reviewer 3

### Summary
The authors observe that models often discover a correct reasoning path early in inference but then perform numerous unnecessary thought switches, undermining reasoning depth and coherence. To address this "under-thinking" problem, they propose Steady Thought (ST), a framework that: (1) segments thoughts using entropy-based detection, (2) completes each thought without further switching, and (3) constructs thought-level preference pairs based on final correctness. Experimental results show that ST successfully mitigates under-thinking by reducing unnecessary switches, leading to more focused reasoning while maintaining or even enhancing performance.

### Strengths
- Clearly identifies and formalizes "under-thinking" as a thought-level preference learning problem, more fine-grained than prior response-level approaches, preserving the model's flexibility to explore alternative reasoning paths when needed.  
- The proposed ST framework combining entropy-based thought segmentation with a SimPO-inspired preference objective (STPO) that effectively mitigates length bias.  
- Strong empirical results: consistent accuracy gains and token reductions across multiple models and datasets, including out-of-distribution generalization to code tasks despite training only on math data.

### Weaknesses
- The reliance on predefined switch tokens (e.g., "wait", "alternatively") limits generalization, especially for models or domains that switch thoughts implicitly without explicit lexical cues. In contrast, concurrent work like SwiReasoning (arXiv:2510.05069) effectively handles implicit thought switching in latent space.  
- Thought segmentation hinges on a tunable entropy threshold; while ablations are provided, its robustness across diverse reasoning styles or model architectures remains unclear.  
- The computational overhead of the ST pipeline, particularly completion per response during data construction, is not adequately discussed.
- The method assumes a correct answer can be derived by completing a single early thought, which may not hold for problems requiring genuine multi-stage exploration or backtracking.  
- The experimental evaluation is limited in scope: only two models and three benchmarks are tested. The paper does not investigate whether the approach scales effectively to larger reasoning models.

### Questions
See the Weaknesses above.

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
4

### Confidence
3

---

## Human Reviewer 4

### Summary
This paper proposes Steady Thought (ST), a thought-level preference optimization framework designed to mitigate the under-thinking phenomenon in large reasoning models (LRMs).

### Strengths
Experimental results are good.

### Weaknesses
1. This work lacks significant novelty and does not offer compelling research insights. The three proposed components share similar ideas with prior studies, making the paper appear more like a compositional work built upon existing methods. Specifically, the designs in Sections 3.1 and 3.2 are rather straightforward, with many previous works employing analogous strategies; thus, the contribution in these parts can hardly be regarded as truly innovative and instead reflects a combination of existing tricks. Furthermore, the main idea of Section 3.3 is almost a direct extension of SimPO, with only minor adjustments in the level of application granularity. It lacks substantial algorithmic innovation and can be viewed as a mild variant rather than a new optimization framework. In summary, this paper represents an incremental improvement at the technical level—although the experimental results are satisfactory, the work as a whole resembles a technical report rather than a research study with strong originality;

2. While threshold tuning is discussed, the paper lacks qualitative or visual evidence (e.g., example trajectories) showing clearer reasoning stabilization;

3. Although the paper reports reduced token counts and small accuracy gains, these improvements could be attributed to shorter decoding or regularization effects rather than the claimed “thought-level optimization.” There is no qualitative or mechanistic evidence showing that the model truly learns a new form of reasoning persistence or gains interpretable control over its thought process.

### Questions
No more.

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
2

### Confidence
4