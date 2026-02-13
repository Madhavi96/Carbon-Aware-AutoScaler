# Carbon-Aware Autoscaling for Performance-Aware Microservices  
## Artifact Package – ICSA 2026 Submission

This repository contains the replication package for the paper:

**Green Autoscaler for Performance-Aware Microservices: A Machine Learning Approach**

---

## Summary

This artifact implements the proposed approach for Kubernetes-based microservice applications.

Unlike traditional autoscalers such as Kubernetes Horizontal Pod Autoscaler (HPA), which rely solely on performance metrics (e.g., CPU utilization), the proposed system integrates these into the autoscaling control loop.

- workload dynamics  
- inter-service dependencies  
- real-time regional carbon intensity data  


Machine learning models based on Spatio-Temporal Graph Convolutional Networks (STGCN) predict pod  counts using monitored system metrics.  
Scaling aggressiveness is dynamically adjusted based on carbon intensity levels to balance:

- response time  
- energy consumption  
- carbon emissions  

The system is evaluated on three benchmark microservice applications [1–3]:

- TrainTicket [1]  
- OnlineBoutique [2]  
- Bookinfo [3]  


## Key Results

Compared to Kubernetes HPA, the proposed autoscaling approach achieves:

- **~24% reduction in carbon emissions** in high carbon-intensity regions (300 gCO₂/kWh)  
- **~17% reduction in carbon emissions** in low carbon-intensity regions (100 gCO₂/kWh)  
- Comparable performance under low-carbon conditions  

Inference overhead remains approximately: ~2.6 kJ per autoscaling cycle


This overhead becomes negligible for medium and large microservice deployments.

---

## Reproducing the Results

### 1. Clone Repository

```bash
git clone https://github.com/Madhavi96/Carbon-Aware-AutoScaler
cd Carbon-Aware-AutoScaler
```

### 2. Deploy the applications, and generate Ground Truth Data

Navigate to the `collect_dataset` directory and run the ground truth generation script for your target application. Replace the <APPNAME> with the desired app name - `trainticket, onlineboutique, bookinfo`.

```bash
cd collect_dataset
bash generate_ground_truth_<APPNAME>.sh
```

### 2. Train the model
```bash
cd ../Deepscaler
python prepareData.py
python main.py
```
### 3. Run inference and evaluation pipeline
```bash
python predict_scale.py
```

### 4. Generate the load to simulate workload

```bash
python predict_scale.py
```

**Benchmark Application References**

[1] Train-Ticket: A Benchmark Microservice System, Fudan SE Lab. Available at: https://github.com/FudanSELab/train-ticket (accessed 13 Feb 2026).

[2] Online Boutique: Cloud-Native Microservices Demo Application, Google Cloud Platform. Available at: https://github.com/GoogleCloudPlatform/microservices-demo (accessed 13 Feb 2026).

[3] Bookinfo Sample Application, Istio. Available at: https://istio.io/latest/docs/examples/bookinfo/ (accessed 13 Feb 2026).

---