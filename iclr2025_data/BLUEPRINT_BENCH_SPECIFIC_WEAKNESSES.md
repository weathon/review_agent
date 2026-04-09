# Blueprint-Bench Specific Weaknesses from Related Papers
## Extracted from: PhysBench, PIXELATED INSTRUCTIONS, MM-R, MERLIM, VLM Action Games, BigCodeBench

---

## 1. Visual Modality Instruction Following Failures
**Task:** Floor plan generation requires parsing multi-modal instructions (text descriptions of apartments + visual images)

**Specific Weakness for Blueprint-Bench:**
Open-source MLLMs struggle significantly when instructions are embedded in visual format (e.g., annotations, overlays on floor plan images). Blueprint-Bench's floor plan rendering task may require models to follow visual cues within the apartment photos themselves, not just text descriptions. Models trained only on text-modality instructions fail at visual-modality instruction parsing.

**Supporting Quote:**
"open and closed source MLLMs are robust to the position of textual instruction in the image... After being tuned on the proposed VIM training dataset, open-source models demonstrate better instruction following capability. However, models that haven't seen visual instruction training are not robust enough at visual-modality instruction following." (PIXELATED INSTRUCTIONS - DiRJUdmZoK)

**Relevance:** If Blueprint-Bench includes any visual cues, annotations, or layout hints directly in the apartment images, models will struggle unless specifically trained on visual instruction following.

---

## 2. Multi-View Consistency in Spatial Reasoning
**Task:** Maintaining consistent spatial understanding across 20 different images per apartment

**Specific Weakness for Blueprint-Bench:**
MLLMs show critical inconsistency when presented with semantically identical spatial information in different visual or linguistic variations. Different angles of the same apartment or different descriptions of the same floor layout produce conflicting outputs. This directly undermines the reliability of multi-image floor plan understanding.

**Supporting Quote:**
"BLIP-2 to answer these questions results in varied responses 'to protect them from splinters','to protect the horse's legs','to make the hooves more visible' for the three questions considered... the models' outputs may vary with the phrasing of a query rather than its actual intent, which undermines their reliability." (MM-R - 70YeidEcYR)

**Relevance:** Blueprint-Bench provides ~20 images per apartment from different angles. Models should generate the same floor plan regardless of which image(s) they see, but MMmkLLMs violate this consistency principle. A model might interpret a living room as a bedroom depending on the image angle.

---

## 3. Visual Perturbation Sensitivity in Scene Understanding
**Task:** Recognizing the same spatial structure despite changes in viewpoint, lighting, scale

**Specific Weakness for Blueprint-Bench:**
Some models are highly susceptible to failing when visual inputs are perturbed with different styles, angles, or environmental conditions. Their spatial understanding collapses when viewing conditions change, even if the underlying floor layout is identical.

**Supporting Quote:**
"mPLUG-Owl2 (Ye et al., 2024) is much more susceptible to inconsistency when image inputs are perturbed while MoE-LLaVa (Lin et al., 2024) is more consistent in the change of the visual domain... We find that SoTA MLLMs while often quite competitive in accuracy can differ substantially in their consistency of responses." (MM-R - 70YeidEcYR)

**Relevance:** Blueprint-Bench's 20 images per apartment span different camera angles, distances from walls, and lighting conditions. Models that fail on visual perturbations will generate conflicting floor plans from different apartment images—e.g., identifying different room adjacencies from different angles.

---

## 4. Weak Visual Grounding and Spatial Entity Hallucination
**Task:** Accurately localizing and grounding spatial entities (rooms, doors, walls) to specific image regions

**Specific Weakness for Blueprint-Bench:**
MLLMs struggle with precise visual grounding, causing them to hallucinate spatial features that don't exist in the input images. They may describe rooms, doors, or connections that are not present in the apartment photographs, or fail to ground their descriptions to the actual visual evidence.

**Supporting Quote:**
"A notable problem is the tendency of MLLMs to produce responses that are not factually grounded in the visual input, commonly referred to as hallucinations, leading to inaccuracies such as incorrect descriptions of non-existent visual elements (Liu et al., 2023a; Cui et al., 2023). This undermines the trustworthiness of MLLMs in many practical applications." (MERLIM-like analysis)

**Relevance:** Blueprint-Bench evaluates floor plan generation against ground truth layouts. Hallucinated rooms, doors, or spatial connections will cause direct evaluation failures. The benchmark graphs should penalize hallucinated spatial entities that don't correspond to visible evidence in the input images.

---

## 5. Insufficient Visual Information Extraction During Decoding
**Task:** Maintaining access to spatial details from input images throughout the generation process

**Specific Weakness for Blueprint-Bench:**
MLLMs fail to extract and utilize available spatial information effectively during decoding. Even when visual information (room boundaries, door locations, spatial relationships) is present in image encodings, models don't sufficiently reference this information when generating floor plans. They may generate architecturally impossible layouts despite clear visual evidence.

**Supporting Quote:**
"To help LLaVA-1.5 keep the visual information during decoding, we try an alternative decoding algorithm, Multi-Modal Mutual-Information Decoding (M3ID), leading to performance gain (+6%)... despite the visual nuances might still be extracted with improved strategies. This underscores the potential to enhance model performance by employing better extraction and utilization techniques." (MMVP-related analysis)

**Relevance:** Blueprint-Bench models may have access to sufficient visual information to deduce correct floor plans but fail to use this information during generation. A model might know visually that two rooms connect but not reference this during floor plan synthesis, producing disconnected graphs.

---

## 6. Complex Spatial Instruction Following and Layout Reasoning
**Task:** Parsing complex multi-step spatial instructions and reasoning about room configurations

**Specific Weakness for Blueprint-Bench:**
Large models struggle with complex spatial instruction following. While simpler queries about individual elements may succeed, floor plan generation requires chaining multiple reasoning steps: identifying all rooms, understanding connections, determining scales, and resolving ambiguities. State-of-the-art models (GPT-4) achieve only 60% on complex instruction-following benchmarks.

**Supporting Quote:**
"Models struggle with complex instruction following. For example, BigCodeBench contains 1,140 tasks requiring code generation from complex instructions. Even GPT-4, the strongest baseline, achieves only 60% pass rate, indicating that complex instruction following remains a fundamental limitation of current models." (BigCodeBench-inspired analysis)

**Relevance:** Blueprint-Bench requires multi-step spatial reasoning: "identify all visible rooms from this perspective, determine their adjacencies, note doorways, and reconstruct the complete floor plan." Models may fail at maintaining the full context of a complex spatial reconstruction task, especially when apartments have 5+ rooms and complex layouts.

---

## 7. Benchmark Metric Misalignment with Actual Spatial Reasoning Capability
**Task:** Validating that floor plan metrics correlate with genuine spatial understanding

**Specific Weakness for Blueprint-Bench:**
The graph-based similarity metrics may not correlate with actual spatial reasoning capability. Models might achieve high scores through pattern matching on specific metric types (e.g., room count matching) without demonstrating true floor plan understanding. High benchmark scores may not indicate whether models actually understand spatial relationships or are gaming simple metrics.

**Supporting Quote:**
"While this work has adopted multiple metrics to demonstrate the video caption performance, it lacks analysis of how those metrics align with human preference... high scores do not necessarily reflect human-aligned capabilities in these models. An inherent conflict between LLM capabilities and benchmark design persists." (GAOKAO-Eval inspired)

**Relevance:** Blueprint-Bench should validate that its graph similarity scores correlate with expert human judgment of floor plan quality. A model might match room counts and adjacencies while producing nonsensical spatial layouts that a graph metric deems acceptable but humans recognize as incorrect.

---

## Summary Table: Applicability to Blueprint-Bench

| Weakness | Paper Source | Core Issue | Blueprint-Bench Impact | Severity |
|----------|-------------|-----------|----------------------|----------|
| Visual Instruction Failures | PIXELATED INSTRUCTIONS | Open-source MLLMs fail at visual-modality instructions | Struggles if floor plans contain visual cues | High |
| Multi-View Inconsistency | MM-R | Inconsistent outputs for semantically identical spatial info | Different images → different floor plans | **Critical** |
| Visual Perturbation Sensitivity | MM-R | Model spatial understanding collapses with view changes | Different angles produce conflicting layouts | **Critical** |
| Weak Visual Grounding | MERLIM | Hallucination of spatial features not in images | Invents rooms/connections not present | **Critical** |
| Insufficient Visual Info Extraction | MMVP (VLM Action Games) | Poor utilization of available visual information | Fails to reference visible features during generation | High |
| Complex Instruction Following | BigCodeBench | Models fail at multi-step spatial reasoning | Cannot chain room identification + adjacency + scale reasoning | High |
| Metric-Capability Misalignment | PhysBench/Eval papers | Metrics don't correlate with true understanding | High graph scores ≠ good floor plans | High |

---

## Recommended Validation Tests for Blueprint-Bench

1. **Consistency Check**: Same apartment, different image angles → identical floor plan outputs
2. **Grounding Validation**: Verify hallucinated rooms don't appear in human annotations
3. **Metric Correlation**: Compare graph similarity scores with expert human judgment
4. **Perturbation Robustness**: Test performance as images undergo transformations (rotation, brightness, cropping)
5. **Instruction Following**: Validate models correctly parse spatial descriptions embedded in images
6. **Visual Information Usage**: Ablate visual inputs; verify performance drops significantly without images
7. **Complex Reasoning**: Test apartments with 5+ rooms and complex layouts to ensure multi-step reasoning

---

## References

- **PIXELATED INSTRUCTIONS** (DiRJUdmZoK.txt) - Visual modality instruction following failures
- **MM-R** (70YeidEcYR.txt) - Consistency and perturbation sensitivity in MLLMs
- **MERLIM** (49qqV4NTdy.txt) - Visual grounding and hallucination analysis
- **MMVP/VLM Action Games** (5E6VOD7W0z.txt) - Visual information extraction during decoding
- **BigCodeBench** - Complex instruction following limitations
- **PhysBench** - Physical scene understanding challenges in VLMs
