### Why Basic Machine Learning Instead of a CNN for SceneMetric

In the SceneMetric project, the goal is to estimate shot scale (CLOSE, MEDIUM, WIDE) using interpretable visual signals extracted from video frames. Instead of using a Convolutional Neural Network (CNN), a lightweight machine learning classifier is used on top of handcrafted computer vision features. This design choice is intentional and provides several advantages.

#### 1. Interpretability

The system extracts meaningful visual features from each frame such as:

* **Face ratio** – proportion of the frame occupied by detected faces
* **Center edge concentration** – how strongly edges cluster near the frame center
* **Depth ratio** – difference in sharpness between center and borders (proxy for depth of field)
* **Subject coverage** – strength of gradients in the central region
* **Entropy** – spatial distribution complexity of edges

Because these features correspond to real cinematographic properties, the model’s decisions can be clearly explained. For example:

* A **high depth ratio** suggests shallow depth of field, which typically indicates a **close shot**.
* **High center edge concentration** indicates the subject occupies the frame center, again suggesting **close framing**.
* **Low center concentration with edges spread across the frame** often corresponds to **wide shots**.

This interpretability allows developers to understand and debug the system easily.

#### 2. Avoiding Black-Box Models

CNNs learn visual patterns automatically but behave largely as **black-box models**. When a CNN predicts a shot scale, it is difficult to determine which visual cues influenced the decision. This makes debugging, analysis, and explanation challenging.

In contrast, the classical computer vision + machine learning approach provides transparency because each decision is derived from explicit features that correspond to cinematic composition.

#### 3. Small Dataset Suitability

Training CNNs typically requires thousands of labeled images to generalize well. In contrast, classical features combined with models such as Random Forests or Gradient Boosting perform well even with relatively small datasets (hundreds of frames).

Since SceneMetric operates on a limited labeled dataset, a lightweight ML classifier is more appropriate and avoids overfitting.

#### 4. Computational Efficiency

Handcrafted feature extraction followed by a small machine learning classifier is computationally lightweight. The system can process frames quickly on a CPU without requiring GPUs or large neural network inference pipelines. This makes the solution practical for experimentation and deployment.

#### 5. Alignment with Classical Computer Vision Goals

SceneMetric is designed as a **classical computer vision analysis system**, where the emphasis is on understanding visual composition rather than simply maximizing classification accuracy. Using explicit features preserves this objective and demonstrates the relationship between cinematography principles and algorithmic analysis.

#### Summary

For SceneMetric, a basic machine learning classifier on top of handcrafted visual features provides:

* transparent and interpretable predictions
* easier debugging and analysis
* better performance on small datasets
* lower computational cost
* alignment with classical computer vision methodology

Therefore, a lightweight machine learning approach is preferred over a CNN-based black-box model for this project.
