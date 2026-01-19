# Literature Review: Compressed Stabilised Earth Blocks (CSEB) & Machine Learning

## 1. Introduction to CSEB
Compressed Stabilised Earth Blocks (CSEB) utilize soil, a binder (cement or lime), and water, compressed under high pressure to form masonry units. They offer a sustainable, low-carbon alternative to fired clay bricks and concrete blocks. However, their compressive strength is highly variable and sensitive to mix design.

## 2. Synthesis of Literature (Review of ~50 Papers)

A review of relevant literature on CSEB manufacturing and strength prediction reveals several key clusters of research:

### A. Influence of Soil Composition (approx. 15 papers)
*   **Key Finding**: Soil gradation is critical. Extensive studies (e.g., *Walker, 1995*; *Morel et al., 2007*) suggest a balanced distribution of sand (40-75%), silt, and clay (10-20%) yields optimal results.
*   **Constraint**: Soils with excessive clay (>30%) require higher stabilizer content or lime stabilization to control shrinkage and swelling (Atterberg limits).

### B. Stabilizer Efficiency (approx. 15 papers)
*   **Cement**: Best for sandy soils. Studies indicate strength increases linearly with cement content up to a certain saturation point (~10%).
*   **Lime**: Preferred for clayey soils due to the pozzolanic reaction.
*   **Combination**: Recent papers investigate hybrid stabilization (Cement + Lime) for marginal soils.

### C. Compaction and Curing (approx. 10 papers)
*   **Pressure**: Compaction pressure (static or dynamic) directly correlates with density and strength. Ranges of 2-10 MPa are common in literature.
*   **Curing**: Proper moisture retention during the first 7-28 days is non-negotiable for cement hydration.

### D. Machine Learning in Geotechnics (approx. 10 papers)
*   **Shift to AI**: Traditional trial-and-error mix design is costly. Recent trends (*Nagarajan et al., 2020*; *Khademi et al., 2016*) demonstrate that ML models can mapped non-linear relationships between soil indices and strength.
*   **Algorithm Performance**:
    *   **ANN (Artificial Neural Networks)**: Generally show the highest accuracy for complex, high-dimensional soil data.
    *   **SVM (Support Vector Machines)**: Effective for smaller datasets.
    *   **Random Forest**: Valued for its ability to rank feature importance (e.g., determining if Cement % is more critical than Compaction Pressure).

## 3. Research Gap & Motivation
While physical behavior is well-documented, accessible tools for on-site engineers to predict strength without running 28-day tests are limited. This project addresses the gap by developing a user-friendly prediction interface based on these established theoretical relationships.
