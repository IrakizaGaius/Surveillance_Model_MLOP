# 🚀 Load Testing Guide for Surveillance Sound API

## 📊 Recommended Load Testing Scenarios

### 1. **Baseline Testing** (Recommended for Start)

```bash
# 10 users over 30 seconds, run for 2 minutes
locust -f locustfile.py --host=http://localhost:8000 \
  --users=10 --spawn-rate=0.33 --run-time=120s \
  --user-class=ModelApiUser
```

### 2. **Normal Load Testing**

```bash
# 50 users over 60 seconds, run for 5 minutes
locust -f locustfile.py --host=http://localhost:8000 \
  --users=50 --spawn-rate=0.83 --run-time=300s \
  --user-class=ModelApiUser
```

### 3. **Peak Load Testing**

```bash
# 100 users over 120 seconds, run for 10 minutes
locust -f locustfile.py --host=http://localhost:8000 \
  --users=100 --spawn-rate=0.83 --run-time=600s \
  --user-class=ModelApiUser
```

### 4. **Stress Testing** (Use HeavyLoadUser)

```bash
# 200 users over 300 seconds, run for 15 minutes
locust -f locustfile.py --host=http://localhost:8000 \
  --users=200 --spawn-rate=0.67 --run-time=900s \
  --user-class=HeavyLoadUser
```

## 🚨 **Why Your Original Config Was Too Aggressive**

### Original: 500 users in 10 seconds (50 users/sec ramp-up)

- **Problem**: Extremely fast ramp-up overwhelms the system
- **Issue**: No time for system to adapt to load
- **Result**: Likely to cause failures and inaccurate results

### Better Approach: Gradual Ramp-up

- **Baseline**: 10 users over 30 seconds (0.33 users/sec)
- **Normal**: 50 users over 60 seconds (0.83 users/sec)
- **Peak**: 100 users over 120 seconds (0.83 users/sec)
- **Stress**: 200 users over 300 seconds (0.67 users/sec)

## 📈 **Expected Performance Metrics**

### **Baseline Load (10 users)**

- **RPS**: 2-5 requests/second
- **Response Time**: <500ms average
- **Error Rate**: <1%
- **CPU Usage**: 10-20%

### **Normal Load (50 users)**

- **RPS**: 8-15 requests/second
- **Response Time**: <800ms average
- **Error Rate**: <2%
- **CPU Usage**: 30-50%

### **Peak Load (100 users)**

- **RPS**: 15-25 requests/second
- **Response Time**: <1.2s average
- **Error Rate**: <5%
- **CPU Usage**: 60-80%

### **Stress Load (200 users)**

- **RPS**: 25-40 requests/second
- **Response Time**: <2s average
- **Error Rate**: <10%
- **CPU Usage**: 80-95%

## 🔧 **Load Testing Commands**

### **Interactive Mode** (Recommended for exploration)

```bash
locust -f locustfile.py --host=http://localhost:8000
```

Then open http://localhost:8089 and configure via web UI.

### **Headless Mode** (For automation)

```bash
# Baseline test
locust -f locustfile.py --host=http://localhost:8000 \
  --users=10 --spawn-rate=0.33 --run-time=120s \
  --headless --html=reports/baseline_test.html

# Normal load test
locust -f locustfile.py --host=http://localhost:8000 \
  --users=50 --spawn-rate=0.83 --run-time=300s \
  --headless --html=reports/normal_load_test.html

# Peak load test
locust -f locustfile.py --host=http://localhost:8000 \
  --users=100 --spawn-rate=0.83 --run-time=600s \
  --headless --html=reports/peak_load_test.html
```

## 📊 **Monitoring During Tests**

### **System Resources**

```bash
# Monitor CPU and Memory
htop

# Monitor network
iftop

# Monitor disk I/O
iotop
```

### **API Metrics**

- Response times
- Error rates
- Throughput (RPS)
- Concurrent users

## 🎯 **Test Scenarios**

### **Scenario 1: Health Check**

- Test `/health` and `/status` endpoints
- Lightweight, fast responses
- Good for baseline performance

### **Scenario 2: Audio Prediction**

- Test `/predict` endpoint with WAV files
- CPU-intensive due to ML inference
- Tests model loading and prediction

### **Scenario 3: Mixed Load**

- Combination of health checks and predictions
- More realistic user behavior
- Tests system under varied load

## 📋 **Pre-Test Checklist**

1. **System Preparation**

   - [ ] API server is running and healthy
   - [ ] Model is loaded and accessible
   - [ ] Test files are available
   - [ ] System resources are adequate
2. **Monitoring Setup**

   - [ ] Resource monitoring tools ready
   - [ ] Logs are being captured
   - [ ] Metrics collection enabled
3. **Test Configuration**

   - [ ] Appropriate user count selected
   - [ ] Realistic ramp-up time set
   - [ ] Test duration planned
   - [ ] Success criteria defined

## 🚨 **Warning Signs During Testing**

### **Immediate Stop Conditions**

- Error rate > 10%
- Response time > 5 seconds
- System becomes unresponsive
- Memory usage > 95%

### **Performance Degradation**

- Response time increasing over time
- Error rate climbing
- CPU usage at 100%
- Memory leaks detected

## 📈 **Post-Test Analysis**

### **Key Metrics to Review**

1. **Response Time Distribution**

   - P50, P90, P95, P99 percentiles
   - Average response time
   - Maximum response time
2. **Throughput**

   - Requests per second
   - Peak RPS achieved
   - Sustained RPS
3. **Error Analysis**

   - Error types and frequencies
   - HTTP status codes
   - Exception details
4. **Resource Utilization**

   - CPU usage patterns
   - Memory consumption
   - Network I/O
   - Disk I/O

## 🔄 **Iterative Testing Process**

1. **Start Small**: Begin with baseline testing
2. **Gradual Increase**: Step up load incrementally
3. **Monitor Closely**: Watch for degradation signs
4. **Document Results**: Record all metrics and observations
5. **Optimize**: Make improvements based on findings
6. **Retest**: Validate improvements with same tests

## 💡 **Tips for Better Load Testing**

1. **Use Realistic Data**: Test with actual audio files
2. **Monitor System Resources**: Don't just focus on API metrics
3. **Test Different Scenarios**: Mix of endpoints and user behaviors
4. **Document Everything**: Keep detailed records of test conditions
5. **Iterate and Improve**: Use results to optimize your system
