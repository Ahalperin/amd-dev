# RCCL Profiling Graphical Visualization Guide

This guide shows you how to view RCCL profiling results graphically using various tools and methods.

## 🎨 Available Visualization Methods

### 1. 🌐 Chrome Tracing (RECOMMENDED)

**Best for**: Timeline visualization, kernel execution flow, API call sequences

**Steps**:
1. Run profiling to generate JSON files:
   ```bash
   /tools/scripts/profile_rccl_basic.sh all_reduce_perf
   ```

2. Open Chrome browser and navigate to:
   ```
   chrome://tracing/
   ```

3. Click "Load" and select your JSON file:
   ```
   /workspace/profiling_results/all_reduce_perf_basic_TIMESTAMP.json
   ```

**Features**:
- ✅ Timeline view of GPU kernels
- ✅ HIP/HSA API call traces  
- ✅ Memory operations
- ✅ Zoom and pan functionality
- ✅ Detailed event information

### 2. 🔥 Perfetto Trace Viewer (ADVANCED)

**Best for**: Advanced timeline analysis, performance metrics

**Steps**:
1. Generate Perfetto traces (when rocprofv3 works):
   ```bash
   /tools/scripts/profile_rccl_advanced.sh all_reduce_perf
   ```

2. Open Perfetto web interface:
   ```
   https://ui.perfetto.dev/
   ```

3. Upload your .pftrace file

**Features**:
- ✅ Advanced timeline visualization
- ✅ Performance counter integration
- ✅ SQL query interface
- ✅ Custom metrics and annotations

### 3. 📊 HTML Visualization (BUILT-IN)

**Best for**: Quick overview, kernel performance analysis

**Steps**:
1. Run profiling:
   ```bash
   /tools/scripts/profile_rccl_basic.sh all_reduce_perf
   ```

2. Create HTML visualization:
   ```bash
   python3 /tools/scripts/create_rccl_visualization.py /workspace/profiling_results/results.csv
   ```

3. Open the generated HTML file in any web browser

**Features**:
- ✅ Top kernel performance chart
- ✅ Detailed kernel table
- ✅ Summary statistics
- ✅ No external dependencies

### 4. 📈 Spreadsheet Analysis

**Best for**: Data analysis, custom charts, statistical analysis

**Steps**:
1. Copy CSV files from container to host:
   ```bash
   docker cp container_id:/workspace/profiling_results/results.csv ./
   ```

2. Open in your preferred spreadsheet application:
   - Microsoft Excel
   - LibreOffice Calc
   - Google Sheets

**Features**:
- ✅ Custom charts and graphs
- ✅ Statistical analysis
- ✅ Data filtering and sorting
- ✅ Export to various formats

### 5. 🐍 Python Analysis (ADVANCED)

**Best for**: Custom analysis, automated reporting, machine learning

**Requirements**: Install pandas, matplotlib, seaborn
```bash
pip install pandas matplotlib seaborn
```

**Example Python script**:
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load profiling data
df = pd.read_csv('/workspace/profiling_results/results.csv')

# Create kernel duration histogram
plt.figure(figsize=(12, 6))
df['Duration_ms'] = df['DurationNs'] / 1000000
plt.hist(df['Duration_ms'], bins=50, alpha=0.7)
plt.xlabel('Kernel Duration (ms)')
plt.ylabel('Frequency')
plt.title('RCCL Kernel Duration Distribution')
plt.show()

# Top 10 kernels by duration
top_kernels = df.nlargest(10, 'Duration_ms')
plt.figure(figsize=(12, 8))
plt.barh(range(len(top_kernels)), top_kernels['Duration_ms'])
plt.yticks(range(len(top_kernels)), top_kernels['KernelName'])
plt.xlabel('Duration (ms)')
plt.title('Top 10 Longest Running Kernels')
plt.tight_layout()
plt.show()
```

## 🚀 Quick Start Examples

### Example 1: Basic Visualization Workflow
```bash
# 1. Run profiling
/tools/scripts/quick_profile_rccl.sh all_reduce_perf

# 2. Create HTML visualization
python3 /tools/scripts/create_rccl_visualization.py /workspace/profiling_results/all_reduce_perf_quick_*.csv

# 3. View in Chrome tracing
# Open chrome://tracing/ and load the .json file
```

### Example 2: Comprehensive Analysis
```bash
# 1. Run detailed profiling
/tools/scripts/profile_rccl_basic.sh all_reduce_perf

# 2. Multiple visualization options:
# - HTML: python3 /tools/scripts/create_rccl_visualization.py results.csv
# - Chrome: Open chrome://tracing/ with .json file
# - Spreadsheet: Copy .csv file to host and open
```

### Example 3: Multi-GPU Visualization
```bash
# 1. Run multi-GPU profiling
/tools/scripts/profile_rccl_multi_gpu.sh all_reduce_perf 2

# 2. Visualize each rank separately
for rank_file in /workspace/profiling_results/*rank*.csv; do
    python3 /tools/scripts/create_rccl_visualization.py "$rank_file"
done
```

## 📋 File Format Guide

### Generated Files and Their Uses:

| File Extension | Best Visualization Tool | Description |
|----------------|------------------------|-------------|
| `.json` | Chrome Tracing | Timeline traces, API calls |
| `.pftrace` | Perfetto | Advanced timeline analysis |
| `.csv` | Spreadsheets, Python | Raw performance data |
| `.db` | SQLite tools | Database queries |
| `.html` | Web Browser | Custom visualizations |

### Key Metrics to Visualize:

1. **Kernel Duration**: Time spent in GPU kernels
2. **Memory Bandwidth**: Data transfer rates
3. **API Overhead**: Time in HIP/HSA calls
4. **GPU Utilization**: Resource usage patterns
5. **Timeline Analysis**: Execution flow and dependencies

## 🔧 Troubleshooting Visualization Issues

### Common Problems:

1. **Large JSON files**: Chrome may struggle with >100MB files
   - **Solution**: Use data filtering or shorter profiling runs

2. **Missing data in visualizations**: Empty or corrupted CSV files
   - **Solution**: Check profiling completed successfully

3. **Browser compatibility**: Some features require modern browsers
   - **Solution**: Use Chrome/Firefox for best compatibility

### Performance Tips:

- **For large datasets**: Use sampling or filter by time ranges
- **For detailed analysis**: Focus on specific kernels or time windows
- **For comparison**: Generate multiple visualizations with consistent parameters

## 📊 Interpretation Guide

### What to Look For:

1. **Long-running kernels**: Potential optimization targets
2. **Memory bottlenecks**: High memory operation times
3. **API overhead**: Excessive time in runtime calls
4. **Load balancing**: Uneven GPU utilization
5. **Communication patterns**: Inter-GPU data transfer efficiency

### Performance Analysis Workflow:

1. **Overview**: Start with HTML visualization for general trends
2. **Timeline**: Use Chrome tracing for execution flow analysis
3. **Deep dive**: Use spreadsheet/Python for statistical analysis
4. **Optimization**: Identify bottlenecks and improvement opportunities

This guide provides multiple pathways to visualize your RCCL profiling data, from quick HTML overviews to advanced timeline analysis tools.
