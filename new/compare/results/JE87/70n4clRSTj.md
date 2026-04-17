# Review

## Summary
This paper introduces SpookyBench, a novel benchmark designed to isolate and evaluate the temporal reasoning capabilities of video-language models (VLMs) by presenting information exclusively through temporal sequences where individual frames appear as noise. The key innovation of SpookyBench lies in its unique design: All meaningful information is encoded exclusively in the temporal domain through dynamic patterns of texts, images, and video depth maps, while individual frames contain only structured noise. The experiments reveal a striking performance gap: while humans effortlessly achieve 98% accuracy on tasks requiring pure temporal pattern recognition, all tested models, including state-of-the-art open and closed-source systems, fail completely with 0% accuracy.

## Soundness
3

## Presentation
3

## Contribution
3

## Strengths
1. The paper is well-written and easy to follow. The authors present their ideas clearly and provide sufficient background information to understand their work.

2. The paper introduces a novel and interesting benchmark that can isolate and evaluate the temporal reasoning capabilities of video-language models. This benchmark has the potential to expose the "time blindness" of current architectures and inspire the development of next-generation temporal-connected models.

3. The experiments are well-designed and provide valuable insights into the temporal reasoning capabilities of both humans and VLMs. The results demonstrate a significant performance gap, highlighting the limitations of current VLMs in processing temporal information.

4. The paper provides a comprehensive analysis of the benchmark, including data statistics, SNR metrics, and the impact of various factors on performance. This analysis helps to understand the properties of the benchmark and the challenges it presents for VLMs.

## Weaknesses
1. The paper does not provide a detailed comparison with other benchmarks that test temporal reasoning capabilities. It would be beneficial to discuss how SpookyBench differs from and improves upon existing benchmarks in this area.

2. The paper does not provide a detailed analysis of the failure modes of the VLMs tested on SpookyBench. Understanding why the models fail could provide valuable insights for improving their temporal reasoning capabilities.

3. The paper does not provide a detailed analysis of the impact of different factors, such as model architecture, training data, and pre-training strategy, on the performance of VLMs on SpookyBench. This could provide valuable insights for improving the temporal reasoning capabilities of VLMs.

## Questions
See weaknesses.

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
6

## Confidence
4