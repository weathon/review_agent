# Extracted Weaknesses Relevant to Blueprint-Bench
## A benchmark for evaluating spatial reasoning in AI models

Blueprint-Bench Context: Tests LLMs, image generation models, and AI agents on floor plan generation from apartment photos using graph-based similarity metrics. Evaluates 50 apartments with ~20 images each. Most models perform at/below random baseline. Identifies instruction following as a problem for image generation models.

---

## 1. Paper: 70YeidEcYR.txt (MM-R: Consistency in MLLMs)

### Weakness 1: Inconsistency Problems in MLLMs

**Concrete Weakness:**
MLLMs show significant inconsistency in their responses to semantically identical queries with visual or linguistic variations, undermining reliability for deployment.

**Quote:**
"This is problematic as the models' outputs may vary with the phrasing of a query rather than its actual intent, which undermines their reliability. Consider the example illustrated in Figure 1 (Mid) top: Most humans would realize that while the three questions (i.e R1, R2, and R3) are superficially different, the semantic meaning is the same. Hence even when the correct answer may not perhaps be apparent (i.e., "to be visible"), the same (consistent) answer should be produced. In contrast, asking models like BLIP-2 (Li et al., 2023b) to answer these questions results in varied responses "to protect them from splinters","to protect the horse's legs","to make the hooves more visible" for the three questions considered."

**Relevance to Blueprint-Bench:**
Blueprint-Bench requires consistent interpretation of floor plans and apartment layouts from multiple images. Inconsistent responses to the same spatial structure described differently would severely impact reliability of the benchmark. Models must maintain consistency across different image representations of the same apartment layout.

---

### Weakness 2: Visual Input Perturbation Sensitivity

**Concrete Weakness:**
Some MLLMs are highly susceptible to inconsistency when visual inputs are perturbed with different styles or modifications.

**Quote:**
"We find that SoTA MLLMs while often quite competitive in accuracy can differ substantially in their consistency of responses. For example, mPLUG-Owl2 (Ye et al., 2024) is much more susceptible to inconsistency when image inputs are perturbed while MoE-LLaVa (Lin et al., 2024) is more consistent in the change of the visual domain than the lingual domain."

**Relevance to Blueprint-Bench:**
Since Blueprint-Bench uses ~20 images per apartment from different angles and lighting conditions, models must handle visual variations without changing their spatial understanding. High sensitivity to image perturbations would cause models to fail at recognizing the same floor plan layout across different photo conditions.

---

## 2. Paper: 49qqV4NTdy.txt (Alignment and Hallucination in Multimodal Models)

### Weakness 3: Hallucination Problems in MLLMs

**Concrete Weakness:**
MLLMs tend to produce responses not factually grounded in visual input, often stating incorrect facts or describing non-existent visual elements, undermining trustworthiness.

**Quote:**
"A notable problem is the tendency of MLLMs to produce responses that are not factually grounded in the visual input, commonly referred to as hallucinations, leading to inaccuracies such as incorrect descriptions of non-existent visual elements (Liu et al., 2023a; Cui et al., 2023). This undermines the trustworthiness of MLLMs in many practical applications."

**Relevance to Blueprint-Bench:**
Blueprint-Bench requires accurate spatial understanding of floor plans and apartment layouts. Hallucination (inventing rooms, doors, or spatial relationships that don't exist) would directly cause floor plan generation errors and incorrect connectivity metrics used to evaluate models.

---

### Weakness 4: Benchmark Limitations in Measuring Hallucinations

**Concrete Weakness:**
Existing hallucination benchmarks have significant limitations in their measurement approaches and scoring, failing to accurately detect certain types of hallucinations.

**Quote:**
"Our study reveals shortcomings in existing benchmarks, particularly around measuring hallucinations... The original MMHALBench benchmark (Sun et al., 2023) uses GPT-4 to judge whether model responses introduce hallucinations. In that text-only regime, MMHALBench relies on ground truth annotations, but found cases where responses with hallucinations were considered as correct. Oppositely, we found cases where valid answers were wrongly tagged as containing hallucinations."

**Relevance to Blueprint-Bench:**
Blueprint-Bench's graph-based similarity metrics may not fully capture all types of spatial reasoning failures (e.g., missing rooms vs. incorrect connections). The benchmark evaluation methodology itself may not accurately measure all failure modes of interest, similar to how existing hallucination benchmarks miss certain types of errors.

---

## 3. Paper: 5E6VOD7W0z.txt (VLM Failures in Visual Reasoning)

### Weakness 5: VLM Failure in Spatial Reasoning Tasks

**Concrete Weakness:**
Vision-Language Models struggle with spatial reasoning tasks, particularly when distinguishing between visually similar images with different spatial relationships.

**Quote:**
"Recent work argued that the pretrained CLIP image encoder (Radford et al., 2021), which serves as the "eyes" of many VLMs, could be the cause and cure for such visual shortcomings (Tong et al., 2024c)... Visually different images could be ambiguously encoded with high cosine similarity in the embedding space. They claimed this suggested information loss and caused VLMs' failure in relevant visual reasoning tasks, such as the MMVP benchmark (Tong et al., 2024c). This benchmark consists of selected, semantically distinct image pairs erroneously agreeing in the CLIP image embedding space, and CLIP-based VLMs failed to answer questions regarding the visual semantic difference better than random chance."

**Relevance to Blueprint-Bench:**
Blueprint-Bench fundamentally tests spatial reasoning (room layouts, connectivity, size relationships). VLM failures at spatial understanding directly translate to poor performance on floor plan generation tasks. The paper shows models perform near random chance on spatial reasoning, a core component of Blueprint-Bench.

---

### Weakness 6: Insufficient Visual Information Utilization

**Concrete Weakness:**
MLLMs fail to extract and utilize available visual information effectively during decoding, even when that information is present in embeddings.

**Quote:**
"To help LLaVA-1.5 keep the visual information during decoding, we try an alternative decoding algorithm, Multi-Modal Mutual-Information Decoding (M3ID) (Favero et al., 2024), leading to performance gain (+6%)... this underscores the potential to enhance model performance by employing better extraction and utilization techniques with the same pretrained image encoder. In conclusion, despite the erroneous agreements in the CLIP embedding space, visual nuances might still be extracted with improved strategies. This underscores the potential to enhance model performance by employing better extraction and utilization techniques with the same pretrained image encoder."

**Relevance to Blueprint-Bench:**
Blueprint-Bench images contain rich spatial information. If models fail to utilize this visual information effectively during inference, they cannot generate accurate floor plans. The problem isn't always that information is unavailable but that models don't extract and use it properly.

---

### Weakness 7: Poor Performance on MMVP Benchmark Remains Unexplained

**Concrete Weakness:**
Even with strong visual information extraction, MLLMs show poor performance on challenging spatial reasoning benchmarks, with the underlying causes remaining unclear.

**Quote:**
"LLaVA-1.5's extracting ability. However, its poor performance on the MMVP benchmark remains a mystery. We look into its failure and provide insight into future directions in the discussion section."

**Relevance to Blueprint-Bench:**
This suggests that Blueprint-Bench may also reveal unexpected failures in spatial reasoning that aren't easily explained by current model understanding. Models may perform poorly for reasons beyond what simple metrics capture.

---

## 4. Paper: 1tZLONFMjm.txt (Benchmark Evaluation - GAOKAO-Eval)

### Weakness 8: Mismatch Between Benchmark Performance and Actual Capabilities

**Concrete Weakness:**
High scores on benchmarks do not necessarily reflect human-aligned capabilities. Models can achieve high scores while struggling with tasks simpler for humans, indicating fundamental inconsistencies in how models approach problems.

**Quote:**
"However, there is a growing concern within the community that LLMs may be "gaming" these benchmarks——achieving high scores while demonstrating instability and unreliability when confronted with tasks that are simple for humans (Zhou et al., 2024). As shown in Figure 1, while LLMs may excel at complex questions, they often struggle with simpler ones. This inconsistency further indicates that LLMs' high score of 90% does not necessarily reflect its ability to handle tasks that are considerably easier for humans."

**Relevance to Blueprint-Bench:**
Blueprint-Bench should be careful that models achieving high scores are actually demonstrating robust spatial reasoning, not gaming simple metrics. Models might score well on graph similarity while failing at basic floor plan interpretation. High benchmark scores may mask fundamental spatial reasoning failures.

---

### Weakness 9: Benchmark Design Limitations After Addressing Data Leakage

**Concrete Weakness:**
Even after addressing data leakage and ensuring comprehensive coverage, an inherent conflict persists between LLM capabilities and benchmark design. High scores do not necessarily reflect human-aligned capabilities.

**Quote:**
"Through the rigorous evaluation process outlined above (see Figure 2), we uncovered a crucial insight: even after mitigating issues such as data leakage and insufficient benchmark coverage, an inherent conflict between LLM capabilities and benchmark design persists, as LLMs continue to exhibit inconsistent performance. Specifically, high scores do not necessarily reflect human-aligned capabilities in these models."

**Relevance to Blueprint-Bench:**
Blueprint-Bench should consider whether its evaluation metrics truly capture spatial reasoning ability or just model pattern-matching on specific metric types. The benchmark may need additional validation to ensure high scores indicate genuine floor plan understanding, not artifact learning.

---

### Weakness 10: Inconsistent Performance Across Similar Difficulty Questions

**Concrete Weakness:**
LLMs exhibit anomalous semi-difficulty-invariant scoring patterns and high variance in performance on questions of similar difficulty, unlike human performance patterns.

**Quote:**
"Further analysis employs the theoretical human performance curve from cognitive psychology, modeled by the Rasch model, to rigorously characterize the deviation of LLM scoring patterns from human performance (Rasch, 1993; Bond & Fox, 2007). This reveals two statistical phenomena: a semi-difficulty-invariant scoring rate and high variance in performance on similarly difficult questions."

**Relevance to Blueprint-Bench:**
Blueprint-Bench apartments of similar complexity should show consistent model performance. If models exhibit high variance on apartments of similar difficulty (e.g., all 2-bedroom layouts), this indicates the model's spatial understanding is unreliable and doesn't scale consistently.

---

## 5. Paper: K4YMFdx2Z2.txt (Unsolvable Problem Detection)

### Weakness 11: Limited Correlation Between Standard Benchmark and Trustworthiness

**Concrete Weakness:**
Performance on standard benchmarks (MMBench) shows little correlation with ability to handle edge cases and unsolvable problems. Community efforts to improve benchmark performance don't contribute to model reliability.

**Quote:**
"The most important finding is that there is little correlation between the performance on the existing MMBench and MM-UPD Bench. This indicates that the community's efforts to improve performance on existing benchmarks do not directly contribute to enhancing model reliability."

**Relevance to Blueprint-Bench:**
Blueprint-Bench should verify that models performing well on floor plan metrics would also handle edge cases robustly (e.g., ambiguous apartments, incomplete images). High performance on basic metrics may not correlate with reliability for challenging cases.

---

### Weakness 12: Benchmark Definition Narrowness and Lack of Diversity

**Concrete Weakness:**
Existing benchmarks lack diversity in their evaluation approaches and provide limited insights into fine-grained model capabilities. They fail to capture the full range of real-world challenges.

**Quote:**
"(i) **The definition of unsolvable problems remains narrow.** Existing benchmarks address only mismatches between images and questions, overlooking other critical challenges such as incomplete or missing answer sets. (ii) **Benchmarks lack diversity and fine-grained analysis.** Existing benchmarks (Guo et al., 2024; Akter et al., 2024; Qian et al., 2024) are built upon conventional benchmarks like VQA v2 (Goyal et al., 2017), COCO (Lin et al., 2014) or cover limited tasks such as spatial reasoning tasks (Akter et al., 2024), suffering from a lack of diversity in their datasets."

**Relevance to Blueprint-Bench:**
Blueprint-Bench uses only 50 apartments. This limited scope may not capture the diversity of real-world floor plans and spatial configurations. The benchmark should consider whether 50 apartments are sufficient to evaluate model generalization across different architectural styles, room types, and layouts.

---

### Weakness 13: Insufficient Evaluation Rigor Across Real-World Scenarios

**Concrete Weakness:**
Existing benchmarks fail to systematically evaluate models across both ideal conditions and realistic failure scenarios, lacking unified metrics that account for trade-offs between answering and knowing when not to answer.

**Quote:**
"(iii) **Rigorous evaluation remains insufficient.** To measure performance in real-world use cases, it is essential to systematically evaluate models both with and without specific instructions tailored for unsolvable problems, but existing work has evaluated only one or the other (Guo et al., 2024; Akter et al., 2024). Furthermore, since there are no unified evaluation metrics that take into account both cases when models should answer (standard) and should not (unsolvable), there are no measures to assess the trade-off between the ability to answer and refrain, which hinders progress in this field."

**Relevance to Blueprint-Bench:**
Blueprint-Bench should evaluate models not just on complete apartments but also on degraded or ambiguous inputs. The benchmark lacks assessment of model reliability when floor plans are unclear or incomplete (e.g., missing rooms, obscured layouts). The benchmark doesn't test whether models know when to abstain from answering.

---

### Weakness 14: Models Struggle with Unsolvable Problems Despite Standard Benchmark Performance

**Concrete Weakness:**
Models demonstrating adequate performance on standard benchmarks (like MMBench) struggle significantly with unsolvable problem detection, showing a notable deficiency in withholding answers when appropriate.

**Quote:**
"Our experiments reveal that even most LMMs, which demonstrate adequate performance on existing benchmarks, struggle significantly with MM-UPD, underscoring a novel aspect of trustworthiness that current benchmarks have overlooked... Most open-source LMMs (Hong et al., 2024; Li et al., 2024a; Xue et al., 2024) achieved less than 10% performance, showing about a 40% gap from GPT-4o (OpenAI, 2024a), without prompts tailored for UPD, despite outperforming closed-source LMMs on MMBench."

**Relevance to Blueprint-Bench:**
Models performing well on Blueprint-Bench metrics may be confabulating floor plans for ambiguous apartments. The benchmark should test whether models properly indicate uncertainty or decline to generate plans when visual information is insufficient or contradictory.

---

## Summary Table

| Paper ID | Weakness Type | Relevance to Blueprint-Bench | Severity |
|----------|---------------|------------------------------|----------|
| 70YeidEcYR | Inconsistency in MLLMs | Models must maintain consistent spatial interpretation across image variations | High |
| 70YeidEcYR | Visual perturbation sensitivity | Different photo angles/lighting may cause different floor plans | High |
| 49qqV4NTdy | Hallucination problems | Models may invent spatial features not present | Critical |
| 49qqV4NTdy | Hallucination benchmark limitations | Evaluation metrics may not capture all spatial errors | High |
| 5E6VOD7W0z | Spatial reasoning failures | Core VLM weakness directly affects floor plan generation | Critical |
| 5E6VOD7W0z | Insufficient visual information utilization | Models don't effectively extract spatial info from images | High |
| 5E6VOD7W0z | Unexplained poor performance on spatial benchmarks | Spatial reasoning failures may have unknown causes | High |
| 1tZLONFMjm | Benchmark-capability mismatch | High scores may not indicate genuine understanding | Medium |
| 1tZLONFMjm | Benchmark design limitations persist | Evaluation metrics may not capture true capability | High |
| 1tZLONFMjm | Inconsistent performance on similar difficulty | Models unreliable on apartments of similar complexity | High |
| K4YMFdx2Z2 | Limited correlation with trustworthiness | Good metrics don't guarantee edge case handling | High |
| K4YMFdx2Z2 | Narrow benchmark definitions | 50 apartments may not represent diversity | Medium |
| K4YMFdx2Z2 | Insufficient real-world evaluation | Missing assessment on degraded/incomplete inputs | High |
| K4YMFdx2Z2 | Models fail unsolvable problem detection | Models confabulate on ambiguous inputs | Critical |
