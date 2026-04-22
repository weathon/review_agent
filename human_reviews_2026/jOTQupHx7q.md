# PropensityBench: Evaluating Latent Safety Risks in Large Language Models via an Agentic Approach

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 4, 6, 4

## Abstract
Recent advances in Large Language Models (LLMs) have sparked concerns over their potential to acquire and misuse dangerous capabilities, posing frontier risks to society. Current safety evaluations primarily test for what a model *can* do—its capabilities—without assessing what it *would* do if endowed with high-risk capabilities. This leaves a critical blind spot: models may strategically conceal capabilities or rapidly acquire them, while harboring latent inclinations toward misuse.  We argue that **propensity**—the likelihood of a model to pursue harmful actions if empowered—is a critical, yet underexplored, axis of safety evaluation. We present 
**PropensityBench**, a novel benchmark framework that assesses the proclivity of models to engage in risky behaviors when equipped with simulated dangerous capabilities using proxy tools. Our framework includes 5,874 scenarios with 6,648 tools spanning four high-risk domains: cybersecurity, self-proliferation, biosecurity, and chemical security. We simulate access to powerful capabilities via a controlled agentic environment and evaluate the models' choices under varying operational pressures that reflect real-world constraints or incentives models may encounter, such as resource scarcity or gaining more autonomy. Across open-source and proprietary frontier models, we uncover alarming signs of propensity: models frequently choose high-risk tools when under pressure, despite lacking the capability to execute such actions unaided. These findings call for a shift from static capability audits toward dynamic propensity assessments as a prerequisite for deploying frontier AI systems safely. Our code and data is available at https://github.com/scaleapi/propensity-evaluation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces PropensityBench, a benchmark measuring LLMs' propensity to use dangerous capabilities when equipped with proxy tools under operational pressure. The framework includes 5,874 scenarios across four domains (self-proliferation, cybersecurity, biosecurity, chemical security) and reveals that even safety-aligned models show significantly increased propensity to choose harmful tools under pressure, with some models reaching 79% failure rates.

### Strengths
**Novel evaluation paradigm**: Shifts from capability assessment to propensity measurement, addressing a critical blind spot in current safety evaluations.

**Comprehensive experimental design**: 5,874 scenarios with 6,648 tools, testing 12 models including frontier systems, with rigorous statistical analysis across multiple pressure levels.

**Important empirical findings**: Reveals shallow alignment where models rely on tool naming rather than consequence reasoning (e.g., O4-mini's propensity jumps from 15.8% to 59.3%).

**Systematic pressure modeling**: Six pressure categories with 12 escalating levels, grounded in human psychology research on decision-making under stress.

### Weaknesses
**Limited ecological validity**: The framework assumes agents "would do" certain actions if empowered, but real-world tool availability, implementation constraints, and deployment contexts remain unknown. The proxy tools in Section 2.2 may not accurately reflect future dangerous capabilities.

**Incomplete domain coverage**: The four domains lack clear selection criteria (Section 2.1). Critical areas like economic manipulation, social-political influence, and physical robotics are absent. Why these specific four? The paper doesn't justify this limitation comprehensively.

**Narrow propensity measurement**: Propensity is reduced to binary tool selection (Section 2.5), missing reasoning traces, hesitation patterns, or self-correction behaviors that might indicate true inclination. Real propensity involves more than final choices.

**Missing defense evaluations**: No testing of existing safety mechanisms (constitutional AI, safety filters, monitoring systems) against the benchmark. How would current defenses perform?

**Static pressure application**: Pressure messages are pre-defined rather than adaptive (Section 2.4), potentially missing how models respond to dynamic, context-aware pressure that adjusts based on their responses.

### Questions
**Domain selection rationale**: What systematic criteria determined these four domains? How would results differ with economic, social, or physical safety domains?

**Propensity measurement validity**: How do you validate that tool selection accurately reflects propensity rather than confusion or misunderstanding? Could you incorporate reasoning analysis?

**Defense mechanism testing**: Which existing safety techniques (RLHF variations, constitutional AI, output filters) have you tested? What mitigation strategies show promise?

**Temporal dynamics**: How does propensity change with repeated exposure to similar scenarios? Do models learn to resist or become more susceptible?

**Cross-domain transfer**: Do models showing high propensity in one domain exhibit similar patterns in others? Is there a general "propensity factor"?

**Human baseline comparison**: How do human participants perform under identical pressure conditions? This would validate the benchmark's psychological grounding.

**Real-world calibration**: How do you ensure proxy tools accurately represent future dangerous capabilities? What validation exists for this assumption?

**Pressure interaction effects**: Do different pressure types interact? For example, does combining time and resource pressure create super-linear effects?

**Model architecture influence**: How do different architectural choices (chain-of-thought, mixture-of-experts) affect propensity independent of scale?

**Deployment implications**: What monitoring infrastructure would be needed to detect propensity signals in production systems before harmful actions occur?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents PropensityBench, a benchmark designed to evaluate preference-aligned behavior in large language models by measuring their response distributions—i.e., propensities—over diverse, socially sensitive decision scenarios. Instead of focusing on single-answer correctness or safety scoring, PropensityBench estimates each model’s response tendency across multiple plausible options, covering areas such as fairness, privacy, morality, and governance.

### Strengths
1. By shifting focus from “right vs. wrong” to distributional preference tendencies, PropensityBench captures alignment subtleties overlooked by existing safety or bias benchmarks.
2. The benchmark covers multiple social and ethical dimensions, includes both open- and closed-weight models, and provides quantitative interpretability through entropy-normalized metrics.

### Weaknesses
1. The benchmark relies on fixed question sets and predefined moral/social contexts. This limits adaptability to evolving social norms or contextual variation across cultures and deployment environments.
2. Since the benchmark measures probabilistic tendencies, model sampling parameters (e.g., temperature, top-p) could strongly influence results, but these effects are not systematically analyzed.

### Questions
1. How robust are the reported alignment propensities to different decoding settings (temperature, nucleus sampling, etc.)?
2. Could PropensityBench be extended with dynamic or user-conditioned scenarios to better capture context-dependent ethical alignment?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces PropensityBench, a novel benchmark designed to evaluate the latent safety risks of Large Language Models (LLMs) by measuring their "propensity" (what they would do) rather than just their "capability" (what they can do). The framework places LLMs in an agentic environment with 5,874 scenarios across four high-risk domains (biosecurity, chemical security, cybersecurity, and self-proliferation). In each scenario, the model must choose between an aligned (safe) tool and a misaligned (harmful) tool to complete a task, while subjected to six different types of escalating operational pressure (e.g., time constraints, resource scarcity). The authors find that while state-of-the-art models exhibit low propensity for harm at "zero pressure," their safety erodes alarmingly when pressure is applied. The results also reveal "shallow alignment" in many models, which rely on explicit tool names (e.g., "harmful") to remain safe and whose propensity for misuse jumps significantly when the harmful tool is given a benign name.

### Strengths
1. High Novelty of the Problem Studied

The paper clearly defines and tackles the critical, underexplored distinction between LLM "capability" (can-do) and "propensity" (would-do). Most existing safety research focuses on whether a model knows how to perform a harmful task (capability). This paper rightly argues that a more immediate risk is whether a model, given the means (via tools or APIs), has the inclination to use them harmfully, especially when incentivized. This is a crucial and novel shift in the evaluation paradigm.

2. Novel and Effective Key Idea

The introduction of "operational pressure" as a core experimental variable is a highly novel and effective methodology. The paper draws inspiration from human psychology, where stress erodes ethical decision-making. Applying this concept to LLMs via six concrete pressure tactics (Time, Financials, Self-Preservation, etc.) provides a realistic "stress test" that uncovers vulnerabilities missed by static benchmarks. The results, showing low propensity at zero pressure and high propensity under stress, validate this approach.

3. Extensive and Solid Benchmark

The paper details a robust, scalable, and transparent two-stage pipeline for generating a large and diverse set of 5,874 scenarios. Creating thousands of realistic scenarios in high-stakes domains is non-trivial. The authors use an LLM-based generation pipeline followed by a rigorous two-stage filtering process. This includes both automated validation and, crucially, review by 54 human experts with domain-specific (e.g., biology, chemistry, CS) academic and professional backgrounds. This ensures high quality and realism.

4. Strong and Actionable Results

The "Tool Naming Sensitivity" test provides a clear, quantifiable measure of "shallow alignment," which is a significant finding. The discovery that a model's safety (e.g., O4-mini) can degrade by 4x simply by changing a tool's name from "harmful" to "benign" (while keeping the consequences identical) is an alarming and important result. It demonstrates that current alignment techniques may be teaching models to simply avoid "scary-sounding" words rather than reasoning about the actual consequences of their actions.

5. Novel Taxonomy for an Emerging Risk

The paper introduces the first systematic taxonomy for the "self-proliferation" risk domain. While cybersecurity and biosecurity are established risk domains, self-proliferation is a more nascent, forward-looking concern. By taking inspiration from biological agents to define seven key risk categories (e.g., "Resource acquisition," "Dominating other agents"), the authors provide a concrete framework for studying a complex and abstract potential risk.

### Weaknesses
1. Confounding Experimental Design Choice

The aligned (safe) tool, $\mathcal{T}^{A}$, is explicitly designed to always return errors, making it impossible to complete the task safely. This setup doesn't test a choice between a working safe path and a working harmful path. Instead, it tests the model's "breaking point" - how much pressure it takes for the model to abandon a failing (safe) path in favor of a working (harmful) one. This conflates "propensity for harm" with "propensity to succeed at all costs." A model might choose $\mathcal{T}^{M}$ not from an inclination to do harm, but because it's the only option offered to fulfill its core objective.

2. Reliance on Simulation Over Execution

The benchmark relies on simulated proxy tools and textual descriptions of consequences rather than a sandboxed environment with real tools. As the authors acknowledge, this is a limitation. A model's decision-making based on a description ("this tool is harmful") may differ from its behavior when actually executing code or interacting with a live system. The benchmark tests reasoning about described actions, not the propensity to act in a more complex, embodied environment.

3. Potential Circularity in Benchmark Generation

The 5,874 scenarios and pressure messages were generated using SOTA LLMs like Gemini-2.5-Pro and OpenAI's O3. The benchmark is being used to evaluate the very models (or their close siblings) that were used to create the test. This introduces a risk of the benchmark inheriting the blind spots, biases, and typical failure modes of the generator models. While the extensive human review mitigates this, it's a potential source of methodological bias.

4. Speculative Nature of the "Self-Proliferation" Domain

The "self-proliferation" domain, while novel, is based on analogies to biological agents, which may not map well to LLMs. This risk domain is far more speculative than the concrete risks in cybersecurity or biosecurity. It's unclear how well these biological analogies (e.g., "survival and legacy preservation," "dominating other agents") map to the actual emergent behaviors of AI. The high failure rates in this domain might reflect the abstract or unusual nature of the tasks rather than a genuine, real-world propensity.

5. Static Application of Pressure

The pressure messages are static and delivered in a fixed, escalating sequence, regardless of the model's responses or reasoning. As the authors note, a more realistic (and likely effective) stress test would involve dynamic pressure that adapts to the model's reasoning. For example, if a model says, "I cannot do this, it is unsafe," a static system just sends the next-level pressure message. A dynamic system could counter the model's specific reasoning, creating a more realistic and challenging evaluation.

### Questions
As suggested by the authors, a powerful next step would be to move at least one domain (like cybersecurity) into a sandboxed environment. This would allow the model to interact with a real (but contained) file system, execute code, and query mock APIs, testing actual behavior rather than reasoning about described behavior. Would it be possible to do this?

### Soundness
3

### Presentation
3

### Contribution
3
