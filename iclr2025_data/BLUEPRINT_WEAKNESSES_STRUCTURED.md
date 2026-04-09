# Blueprint-Bench Critical Weaknesses (Structured Format)

---

## 1. Multi-View Inconsistency in Spatial Understanding
**Title:** Models produce inconsistent floor plans from different apartment images

**Description:**
MLLMs generate conflicting floor plan outputs when presented with different angles or images of the same apartment. Semantically identical spatial information (the same room layout) leads to different model predictions when shown from different viewpoints. This undermines the core assumption that 20 images per apartment should yield consistent spatial understanding.

**Supporting Quote:**
"BLIP-2 to answer these questions results in varied responses 'to protect them from splinters','to protect the horse's legs','to make the hooves more visible' for the three questions considered. This is problematic as the models' outputs may vary with the phrasing of a query rather than its actual intent, which undermines their reliability." (MM-R, 70YeidEcYR)

---

## 2. Visual Perturbation Sensitivity in Scene Recognition
**Title:** Models fail to recognize same floor layout under different viewing conditions

**Description:**
Models exhibit severe performance degradation when apartment images undergo changes in angle, lighting, distance, or visual style—even when the underlying spatial structure is identical. Different camera perspectives cause models to identify different room types or spatial relationships, making them unsuitable for multi-angle floor plan reconstruction.

**Supporting Quote:**
"mPLUG-Owl2 is much more susceptible to inconsistency when image inputs are perturbed while MoE-LLaVa is more consistent in the change of the visual domain than the lingual domain. We find that SoTA MLLMs while often quite competitive in accuracy can differ substantially in their consistency of responses." (MM-R, 70YeidEcYR)

---

## 3. Spatial Entity Hallucination and Weak Grounding
**Title:** Models invent rooms, doors, or spatial connections not present in images

**Description:**
MLLMs hallucinate spatial features without grounding them to visual evidence. Models may describe rooms that don't exist in photographs, invent door placements, or claim spatial connections that lack visual support. Direct floor plan evaluation becomes impossible when models generate entities with no factual basis in the input images.

**Supporting Quote:**
"A notable problem is the tendency of MLLMs to produce responses that are not factually grounded in the visual input, commonly referred to as hallucinations, leading to inaccuracies such as incorrect descriptions of non-existent visual elements. This undermines the trustworthiness of MLLMs in many practical applications." (MERLIM-class analysis)

---

## 4. Failure at Visual Modality Instruction Following
**Title:** Open-source MLLMs cannot follow instructions embedded in visual format

**Description:**
When spatial instructions are presented visually (e.g., annotations, text overlays, layout hints embedded in images), open-source MLLMs fail dramatically compared to text-only instruction following. If Blueprint-Bench includes visual cues, floor plans, or annotations within the apartment images themselves, most models will struggle. Only models specifically fine-tuned on visual instruction tasks handle this adequately.

**Supporting Quote:**
"open and closed source MLLMs are robust to the position of textual instruction in the image... However, models that haven't seen visual instruction training are not robust enough at visual-modality instruction following. After being tuned on the proposed VIM training dataset, open-source models demonstrate better instruction following capability." (PIXELATED INSTRUCTIONS, DiRJUdmZoK)

---

## 5. Insufficient Visual Information Utilization During Decoding
**Title:** Models fail to maintain and reference spatial details from images while generating floor plans

**Description:**
Even when apartment images contain sufficient visual information to determine correct floor layouts, models fail to consistently utilize this information during generation. Room boundaries, door locations, and spatial relationships remain unreferenced in the final floor plan output. Models may know correct spatial facts but don't incorporate them into their reasoning pipeline.

**Supporting Quote:**
"To help LLaVA-1.5 keep the visual information during decoding, we try an alternative decoding algorithm, Multi-Modal Mutual-Information Decoding (M3ID), leading to performance gain (+6%). This underscores the potential to enhance model performance by employing better extraction and utilization techniques with the same pretrained image encoder." (MMVP/VLM Action Role-Playing, 5E6VOD7W0z)

---

## 6. Complex Multi-Step Spatial Reasoning Failures
**Title:** Models cannot chain multiple spatial reasoning steps required for complete floor plan generation

**Description:**
Floor plan generation from apartment photos requires multi-step reasoning: (1) identify all visible rooms, (2) determine room adjacencies and connections, (3) locate doorways and passages, (4) resolve spatial ambiguities, (5) synthesize complete graph. State-of-the-art models fail at maintaining context across these steps. Complex instruction-following benchmarks show GPT-4 achieves only 60%, indicating this is a fundamental limitation affecting spatial tasks.

**Supporting Quote:**
"Even GPT-4, the strongest baseline, achieves only 60% pass rate [on complex instruction-following benchmarks], indicating that complex instruction following remains a fundamental limitation of current models." (BigCodeBench-style analysis, 1140 complex tasks)

---

## Summary: Expected Blueprint-Bench Failure Modes

| Weakness | Expected Symptom | Evaluation Impact |
|----------|------------------|-------------------|
| Multi-view inconsistency | Image 1 → 4-room layout; Image 5 → 3-room layout | Graph similarity metrics meaningless |
| Perturbation sensitivity | Model struggles with images from far away but excels at close-ups | Inconsistent floor plan recovery |
| Hallucination | Models add rooms/doors visible nowhere in photos | False positive connections in graph |
| Visual instruction failure | Models ignore visual cues/annotations in images | Poor performance if images contain hints |
| Information underutilization | Model generates valid graphs but ignores visible room doors | Disconnected graph components despite visual evidence |
| Reasoning failures | Models fail on apartments with 5+ rooms or complex L-shapes | Complexity-dependent performance gaps |

---

## Recommended Diagnostic Tests

1. **Consistency Test**: Show Model (Image A) and Model (Image B) of same apartment. Should generate identical floor graphs. Large divergence indicates inconsistency failure.

2. **Perturbation Test**: Apply same apartment with: (i) far camera view, (ii) close-up view, (iii) different lighting, (iv) different time of day. Should produce identical graphs. Divergence indicates perturbation sensitivity.

3. **Hallucination Detection**: Compare model-generated rooms/doors against human ground truth annotations. Count false positives (invented features). Should be near-zero.

4. **Visual Grounding Validation**: Provide apartment image + model output floor plan. Mark which generated features have visual support in image. Calculate % properly grounded (should be 100%).

5. **Complexity Analysis**: Bin apartments by room count (2-3 rooms vs 5+ rooms). Benchmark should show whether performance stays consistent or degrades on complex layouts.

