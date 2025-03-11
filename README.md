# Carbon Aware AutoScaler for Scaling Microservices based Applications in cloud

### Project Overview
- With the rapid growth of microservices-based applications in the cloud, the demand for scalable computing resources has led to increased energy consumption and carbon emissions.
- Traditional approaches that limit computing resources during high carbon intensity periods often compromise application performance, particularly response times.
- This project aims to build and evaluate a machine learning-based carbon-aware autoscaler that addresses the tradeoff between sustainability (energy consumption) and performance ().

### Key Objectives
- Incorporate Carbon Awareness: Use carbon intensity data to guide scaling decisions, aligning cloud operations with sustainability goals.
- Balance Performance & Energy Trade-off: Maintain critical performance metrics (e.g., response time) while reducing carbon emissions.
- Integrate Machine Learning: Explore ML-based autoscaling techniques to enhance energy efficiency without significant performance degradation.
This solution aims to improve the sustainability of cloud-native applications while ensuring optimal performance. 🚀

### Methodology

- Inspired by the DeepScaler [https://github.com/SYSU-Workflow-Lab/DeepScaler] project, we have developed a novel mechanism to take into account the carbon emission in scaling decisions, and make better scaling decisions which are carbon-aware and have least impact to the application response time.


- Proposed Pipeline

<img width="562" alt="image" src="https://github.com/user-attachments/assets/59b7354b-fa81-4d31-bb08-ab8a3f3cc803" />

- Model Architecture
<img width="313" alt="image" src="https://github.com/user-attachments/assets/d583f20d-bed8-40ae-97bf-70ec52a9dc34" />


### Evaluation Results
 - Response Time comaprison 
<img width="450" alt="image" src="https://github.com/user-attachments/assets/59101ffa-82a6-484a-9b20-b699d3039919" />

- Carbon footprint comparison
<img width="455" alt="image" src="https://github.com/user-attachments/assets/829b59de-a151-492c-84c2-2c30def79a59" />
