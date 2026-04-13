=== CALIBRATION EXAMPLE 5 ===

# Final Consolidated Review
## Summary
This paper introduces **Visual Modality Instruction (VIM)**, a benchmark setting in which the task instruction is rendered inside the image rather than supplied as a separate text prompt, and uses it to test whether MLLMs can follow visually embedded instructions. Across eight benchmarks, the paper shows a striking and consistent gap between standard text-instruction evaluation (TEM) and VIM for several open-source MLLMs, and then trains **v-MLLM** by converting existing instruction-tuning data into the same VIM format, substantially improving performance in the VIM setting while mostly preserving TEM performance.

## Strengths
- **The paper exposes a concrete and important failure mode that standard MLLM evaluation largely hides.** Table 3 shows extremely large TEM→VIM drops for several open-source systems, often to near-zero on multiple tasks (e.g., LLaVA-1.5-7b on OKVQA: 58.41 → 0; TextVQA: 45.36 → 0; ChartQA: 18.08 → 0). This is a specific and nontrivial empirical finding, not a generic “models are brittle” claim.
- **The evaluation is broad enough to establish that the phenomenon is not benchmark-specific.** The paper adapts VIM to eight heterogeneous benchmarks (MME, MM-Vet, OKVQA, VizWiz, TextVQA, MathVista, ChartQA, MMMU) and evaluates a meaningful spread of proprietary and open-source models. The pattern is consistent across tasks and model families, which strengthens the core observation.
- **The Mix Instruction ablation is genuinely informative.** Table 5 shows that adding a minimal text prompt like “Answer the question in the image” recovers a large amount of performance for some models (e.g., Qwen-VL-Chat on OKVQA: 0.01 → 30.75; InstructBLIP on VizWiz: 0 → 18.79). This supports the paper’s claim that a key failure is not only answering the question, but recognizing and operationalizing the embedded text as an instruction.
- **The proposed training intervention is simple but effective.** v-MLLM is trained by augmenting the instruction-tuning corpus with VIM-formatted examples, and Table 7 shows substantial VIM gains over LLaVA while keeping TEM performance reasonably close. For example, v-MLLM-13b on OKVQA moves from 0.38 to 54.76 in VIM while remaining near the LLaVA TEM score (59.37 vs. 61.27).
- **The paper includes at least some mechanism-oriented analysis rather than only benchmark tables.** Section 4 decomposes the issue into instruction recognition and instruction following, and Table 4 / Figure 5 provide evidence that some models can describe the image or partially read the text without faithfully extracting the intended instruction semantics.

## Weaknesses

### Fatal
- None.

### Major:
- **The benchmark does not cleanly isolate “instruction-following robustness” from OCR / rendering / visual preprocessing effects.**  
  This is the main concern. The paper’s central framing is about whether MLLMs can follow instructions when those instructions are “in pixels,” but the VIM transformation changes more than modality: it appends rendered text, adds whitespace, changes aspect ratio/canvas size, and routes the instruction through each model’s visual pipeline and resizing strategy. The paper itself acknowledges that “The resolution of the image is also the key” (Sec. 2.1.2) and that VIM requires “strong visual interpretation capability to recognize and follow the embedded instruction.” Section 4.1 further shows instruction recognition failures. So the measured gap is real, but the paper often interprets it too broadly as a failure of instruction following, when the construct being tested is a mixture of OCR/readability, visual encoding robustness, and instruction use. For a benchmark paper, this construct-validity issue matters because it weakens the precision of the headline claim.
- **The generalization claims for v-MLLM are overstated relative to the evidence.**  
  The abstract calls v-MLLM “a generalizable model,” and the paper repeatedly describes it as robust across modalities. However, the training procedure in Sec. 2.2 creates VIM examples by converting the original instruction corpus into essentially the same format used at evaluation: first-turn instruction rendered into the image, with a largely fixed style and default bottom placement. This shows that training on this transformation family helps on this transformation family, which is useful, but it does not by itself demonstrate broader generalization to other visually embedded instruction formats. The paper does not provide cross-format tests such as different fonts, layouts, overlays, cluttered scenes, screenshots, or naturally occurring document/UI settings.
- **The mechanistic evidence for the paper’s recognition-vs-following story is too limited.**  
  The decomposition in Section 4 is promising, but the strongest direct evidence is small: Figure 5 uses only 30 manually checked VQAv2 samples, in an “ideal setup,” and only for a few models. That is too thin to support strong conclusions like “GPT-4V can recognize the embedded instructions nearly perfectly” or to explain the dramatic failures across eight benchmarks. The Mix Instruction results are useful, but they do not fully disentangle whether the bottleneck is OCR, semantic parsing of the recognized text, task framing, or downstream reasoning.
- **Rendering choices are central to the task definition but are only weakly studied.**  
  Section 2.1.2 examines instruction location only on a 21-example MM-Vet subset, then fixes bottom placement. Image resolution is discussed qualitatively rather than systematically, despite being explicitly acknowledged as important. The paper does not systematically vary font size, font family, contrast, line wrapping, text length, or clutter/background interaction. Because these choices define the effective difficulty of VIM, the limited analysis leaves open how much of the reported phenomenon is specific to one synthetic rendering pipeline.
- **The empirical support for “robust in both TEM and VIM” should be stated more carefully.**  
  v-MLLM clearly helps on VIM, but it does not eliminate the gap, especially on more text-heavy or reasoning-heavy tasks. Examples from Table 3 include MathVista (v-MLLM-13b: 28.2 TEM vs. 9.5 VIM) and MMMU (35.4 vs. 29.9), and even on stronger tasks the VIM degradation remains substantial. So the method is a meaningful improvement, but the paper’s language sometimes implies a stronger level of robustness than the numbers support.

### Minor
- **The paper would benefit from a stronger baseline that separates “better OCR” from “better end-to-end VIM reasoning.”**  
  A simple OCR-plus-standard-MLLM pipeline is an obvious comparison and would be highly informative. Without it, it remains unclear whether v-MLLM’s gains primarily reflect implicit OCR adaptation versus a deeper improvement in handling instruction-as-image end-to-end.
- **The ecological validity of the current VIM construction is only partial.**  
  The setting of appending the instruction under the image is a useful probe, but it is narrower than the title’s broader framing of “printed instructions in images.” The paper does not test naturally occurring settings such as UI screenshots, forms, slides, or document-style layouts where visual instructions arise more organically.
- **Some side discussions reduce confidence rather than helping the argument.**  
  In Section 3.4 / Table 3 the paper notes that pure LLMs can score nontrivially in TEM and remarks that Llama 2 even exceeds GPT-4 on several tasks, but then defers the discussion. Since this is surprising and interacts with the paper’s motivation about text-channel priors, the treatment feels underdeveloped.

### Trivial
- None.

## Nice-to-Haves
- Add cross-format generalization tests: train on one rendering style and evaluate on different positions, fonts, contrasts, wrapping rules, and cluttered overlays.
- Add a larger-scale recognition study with automated metrics and a clearer failure taxonomy: cannot read text / reads text but misparses instruction / follows instruction but answers incorrectly.
- Include a simple OCR→text-prompt baseline to show whether end-to-end v-MLLM provides value beyond explicit text extraction.
- Break down performance by instruction length, question type, and benchmark subtype to better characterize what makes VIM hard.
- Evaluate on more naturally occurring visual-instruction scenarios to support the broader motivation.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“Need missing related work X/Y/Z.”** Removed per instructions; external coverage cannot be verified here.
- **Complaints about cited models/tools/benchmarks possibly not existing, being unreleased, or unverifiable.** Removed by rule.
- **Pure formatting/style complaints.** The extracted text has parser artifacts; these are not valid paper weaknesses.
- **Claims that the paper lacks every training hyperparameter or implementation detail.** The paper does provide the main training setup (dataset source, model family, epochs, batch sizes, hardware, stage-wise vs. mixture), and demanding exhaustive artifact-level details is not a substantive review point here.
- **Any criticism that the benchmark is unfair because baselines are weaker in OCR or visual resolution by design.** The asymmetry does not favor the proposed method in the main benchmark comparison; the benchmark’s purpose is precisely to probe this weakness. The valid concern is construct validity and interpretation, not “unfairness” per se.
- **Overstated claim that the work is “not even a paper” or fundamentally invalid.** The paper clearly contains a real empirical phenomenon, a benchmark adaptation, and a working training intervention; the issues are about framing, construct validity, and depth of analysis, not absence of contribution.

## Novel Insights
The most important synthesis across the reviews is that the paper’s strongest contribution is not “solving visual instruction following” but exposing a **privileged text-channel dependency** in many open-source MLLMs: they perform well when the instruction is delivered through the language pathway, but can collapse when the same instruction must be recovered and interpreted through the visual pathway. The Mix condition sharpens this insight: many models improve substantially once a tiny text prompt tells them that the image contains an instruction, suggesting the bottleneck is partly about converting visually detected text into an executable task representation, not just about answering the downstream question. This makes the work valuable as a diagnostic of modality asymmetry, even if the current VIM instantiation still conflates that asymmetry with OCR/rendering effects.

## Suggestions
- Tighten the central claim: present VIM as a probe of **visualized-instruction robustness** rather than a clean measure of abstract instruction following.
- Add an OCR-pipeline baseline and explicitly compare it against v-MLLM.
- Scale up Section 4.1 substantially and turn it into a systematic error analysis.
- Test out-of-distribution rendering formats to justify the words “generalizable” and “robust.”
- Analyze remaining TEM↔VIM tradeoffs more candidly, especially on MathVista / MMMU / ChartQA.
- Narrow the title/abstract wording if additional real-world or cross-format evidence cannot be added.

# Actual Human Scores
Individual reviewer scores: [3.0, 5.0, 3.0, 5.0]
Average score: 4.0
Binary outcome: Reject
