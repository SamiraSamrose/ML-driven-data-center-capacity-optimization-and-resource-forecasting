## System Design Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA SOURCES LAYER                                │
├─────────────┬─────────────┬─────────────┬─────────────┬─────────────────────┤
│   Cluster   │    Power    │   Network   │   Storage   │   Cloud Resources   │
│   Metrics   │Consumption  │   Traffic   │    I/O      │    Utilization      │
│  (Azure)    │    (UCI)    │  (Synth)    │  (Synth)    │     (Synth)         │
│ CPU/Memory  │ IT/Cooling  │ Bytes/Pkts  │IOPS/Latency │ Inst/CPU/Cost       │
└──────┬──────┴──────┬──────┴──────┬──────┴──────┬──────┴──────┬──────────────┘
       │             │             │             │             │
       └─────────────┴─────────────┴─────────────┴─────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   DATA INGESTION LAYER      │
                    │  (Apache Flume Pattern)     │
                    ├─────────────────────────────┤
                    │ • FlumeSource (batch read)  │
                    │ • Event wrapping            │
                    │ • 5 parallel sources        │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   INTERCEPTOR CHAIN         │
                    ├─────────────────────────────┤
                    │ • Timestamp enrichment      │
                    │ • Host identification       │
                    │ • Static tag injection      │
                    │ • Regex filtering (alerts)  │
                    └──────────────┬──────────────┘
                                   │
              ┌────────────────────┴────────────────────┐
              │                                         │
    ┌─────────▼─────────┐                    ┌──────────▼─────────┐
    │  MEMORY CHANNEL   │                    │   FILE CHANNEL     │
    │  (10K capacity)   │                    │  (50K capacity)    │
    │  Fast access      │                    │  Large buffer      │
    └─────────┬─────────┘                    └──────────┬─────────┘
              │                                         │
              └────────────────────┬────────────────────┘
                                   │
           ┌───────────────────────┴───────────────────────┐
           │              PREPROCESSING LAYER              │
           ├───────────────────────────────────────────────┤
           │ • Feature Engineering (temporal, rolling)     │
           │ • Imputation (forward/backward/zero fill)     │
           │ • Normalization (StandardScaler)              │
           │ • Derived metrics calculation                 │
           └───────────────────────┬───────────────────────┘
                                   │
        ┌──────────────────────────┴──────────────────────-────┐
        │                                                      │
┌───────▼─────────┐                              ┌─────────────▼─────────┐
│  STATISTICAL    │                              │    ML TRAINING        │
│  ANALYSIS       │                              │    PIPELINE           │
├─────────────────┤                              ├───────────────────────┤
│ • Descriptive   │                              │ • 7 Regression Models │
│ • Normality     │                              │   - Linear/Ridge      │
│ • Hypothesis    │                              │   - Random Forest     │
│ • Correlation   │                              │   - XGBoost/LightGBM  │
│ • Outliers      │                              │   - Neural Network    │
└────────┬────────┘                              │ • 80/20 Train/Test    │
         │                                       │ • Metrics: MAE/RMSE/R²│
         │                                       └──────────┬────────────┘
         │                                                  │
         │                  ┌───────────────────────────────┘
         │                  │
         │         ┌────────▼────────┐
         │         │ TIME SERIES     │
         │         │ FORECASTING     │
         │         ├─────────────────┤
         │         │ • ARIMA (5,1,2) │
         │         │ • Prophet       │
         │         │ • Exp Smoothing │
         │         │ • LSTM (24 lag) │
         │         │ • MAPE < 15%    │
         │         └────────┬────────┘
         │                  │
         └─────────┬────────┘
                   │
    ┌──────────────▼──────────────┐
    │   ANALYTICS & OPTIMIZATION  │
    ├─────────────────────────────┤
    │ • K-Means Clustering        │
    │ • Isolation Forest Anomaly  │
    │ • L-BFGS-B Optimization     │
    │ • Capacity Planning         │
    │ • Cost Strategy Analysis    │
    └──────────────┬──────────────┘
                   │
         ┌─────────┴─────────┐
         │                   │
┌────────▼────────┐  ┌───────▼────────┐
│ MODEL DEPLOYMENT│  │ DATA CENTER    │
│ INFRASTRUCTURE  │  │ ARCHITECTURE   │
├─────────────────┤  ├────────────────┤
│ • Model Serving │  │ • Rack Design  │
│ • Latency Test  │  │ • Network Topo │
│ • Residual Diag │  │ • Power Calc   │
│ • Autoscaling   │  │ • Cooling Sys  │
│ • P99 < 10ms    │  │ • TCO Model    │
└────────┬────────┘  └───────┬────────┘
         │                   │
         └─────────┬─────────┘
                   │
    ┌──────────────▼──────────────┐
    │    SINK & OUTPUT LAYER      │
    ├─────────────────────────────┤
    │ • HDFS Sink (timestamp path)│
    │ • Kafka Sink (partitioned)  │
    │ • Elasticsearch (indexed)   │
    │ • Logger (log-level)        │
    └──────────────┬──────────────┘
                   │
         ┌─────────┴─────────┐
         │                   │
┌────────▼────────┐  ┌───────▼────────┐
│  VISUALIZATION  │  │   REPORTING    │
│     LAYER       │  │     LAYER      │
├─────────────────┤  ├────────────────┤
│ • 13 Static PNG │  │ • 13 Sections  │
│ • 2 Interactive │  │ • Exec Summary │
│ • Distributions │  │ • Tech Details │
│ • Time Series   │  │ • Recommend    │
│ • Correlations  │  │ • Trade-offs   │
│ • Residuals     │  │ • Benchmarks   │
│ • 3D Scatter    │  │ • Troubleshoot │
└─────────────────┘  └────────────────┘


DATA FLOW:
Raw Data → Ingestion → Interceptors → Channels → Preprocessing → 
[Statistical Analysis + ML Training + Time Series] → 
Analytics/Optimization → [Deployment + Architecture] → 
Sinks → [Visualization + Reporting]

KEY METRICS:
• Throughput: 5K+ events/sec
• Inference: <5ms mean, <10ms P99
• Forecast: MAPE <15%
• Cost Reduction: 35% potential
• Data Volume: 100K+ records
• Models: 7 regression, 4 time series
• Clusters: Optimal K via silhouette
• Anomalies: 5% detection rate
• TCO: 5-year calculation
• Network: 8 Tbps bisection BW
• PUE: 1.5 target
```

**Component Interactions:**

1. **Ingestion → Preprocessing**: FlumeSource reads batches → Interceptors enrich → Channels buffer → Feature engineering transforms
2. **Preprocessing → Analysis**: Normalized data feeds statistical tests and ML pipelines in parallel
3. **Analysis → Optimization**: Model predictions inform optimization constraints and capacity planning
4. **Optimization → Deployment**: Optimal parameters configure serving infrastructure and autoscaling
5. **All Components → Sinks**: Results written to HDFS/Kafka/ES/Logs for persistence
6. **Sinks → Visualization/Reporting**: Aggregated data rendered as charts and comprehensive report

**Concurrency Model**: Parallel processing across statistical, ML, time series modules; sequential within each pipeline stage

**Storage Architecture**: In-memory processing with sink-based persistence to distributed storage systems

**Scalability Points**: Source parallelization, channel capacity expansion, sink replication, model serving autoscaling