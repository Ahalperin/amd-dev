# 🚀 START HERE - RCCL Research & Optimization Guide

**Welcome!** This directory contains comprehensive documentation for RCCL performance analysis and optimization.

---

## 📍 You Are Here

```
/Users/ahalperin/xai/amd-dev/tools/docs/rccl/
```

---

## 🎯 Quick Decision Guide

### I want to...

**→ Understand RCCL architecture**
- Start with: [rccl-design-overview.md](rccl-design-overview.md)
- Time: 1-2 hours
- Level: Beginner

**→ Find and fix performance bottlenecks**
- Start with: [rccl-bottleneck-analysis.md](rccl-bottleneck-analysis.md)
- Time: 2-3 hours
- Level: Intermediate

**→ Understand code implementation**
- Start with: [rccl-technical-internals.md](rccl-technical-internals.md)
- Time: 3-4 hours
- Level: Advanced

**→ Get quick answers while debugging**
- Start with: [quick-reference.md](quick-reference.md)
- Time: As needed
- Level: All

**→ Plan systematic optimization**
- Start with: [optimization-roadmap.md](optimization-roadmap.md)
- Time: 1 hour
- Level: Intermediate

**→ Navigate all documentation**
- Start with: [README.md](README.md) or [INDEX.md](INDEX.md)
- Time: 30 min
- Level: All

**→ Configure RCCL with environment variables**
- Start with: [rccl-environment-variables-analysis.md](rccl-environment-variables-analysis.md)
- Time: 1 hour
- Level: Intermediate

---

## 📚 Complete Document List

| # | Document | Lines | Size | Purpose |
|---|----------|-------|------|---------|
| 1 | [README.md](README.md) | 551 | 19K | Navigation hub |
| 2 | [INDEX.md](INDEX.md) | 565 | 18K | Complete index |
| 3 | [rccl-design-overview.md](rccl-design-overview.md) | 799 | 25K | Architecture guide |
| 4 | [rccl-bottleneck-analysis.md](rccl-bottleneck-analysis.md) | 970 | 24K | Optimization guide |
| 5 | [rccl-technical-internals.md](rccl-technical-internals.md) | 1,210 | 30K | Implementation details |
| 6 | [quick-reference.md](quick-reference.md) | 522 | 12K | Quick lookup |
| 7 | [optimization-roadmap.md](optimization-roadmap.md) | 613 | 15K | 12-week plan |
| 8 | [rccl-environment-variables-analysis.md](rccl-environment-variables-analysis.md) | 571 | 29K | Env variables |
| 9 | [rccl-branch-analysis.md](rccl-branch-analysis.md) | 258 | 9K | Branch comparison |

**Total: 6,279 lines, ~180 KB**

---

## ⚡ 5-Minute Quickstart

```bash
# 1. Run your first test
cd /Users/ahalperin/xai/amd-dev/amd/rccl-tests
./build/all_reduce_perf -b 128M -e 128M -g 8 -n 100

# 2. Check the "busbw" column
# - MI300X target: 350-400 GB/s
# - MI250X target: 240-270 GB/s

# 3. If performance is low, enable debug logging
export NCCL_DEBUG=INFO
./build/all_reduce_perf -g 8 2>&1 | grep -E "xGMI|Channel|Algo"

# 4. Read the appropriate guide based on what you find
```

---

## 🗺️ Learning Path

### Path 1: Performance Engineer (5-8 hours)
```
README.md (30 min)
    ↓
rccl-design-overview.md (1-2 hr)
    ↓
rccl-bottleneck-analysis.md (2-3 hr)
    ↓
optimization-roadmap.md (1 hr)
    ↓
Start optimizing! (Use quick-reference.md as needed)
```

### Path 2: RCCL Developer (7-11 hours)
```
README.md (30 min)
    ↓
rccl-design-overview.md (1-2 hr)
    ↓
rccl-technical-internals.md (3-4 hr)
    ↓
rccl-bottleneck-analysis.md (2-3 hr)
    ↓
Start coding! (Use quick-reference.md as needed)
```

### Path 3: Quick Start (1-2 hours)
```
quick-reference.md (30 min)
    ↓
Run tests and profile
    ↓
Read relevant sections from other docs as needed
```

---

## 🎓 What You'll Learn

- ✅ How RCCL works (architecture, algorithms, protocols)
- ✅ How to profile and identify bottlenecks
- ✅ How to optimize for 2x performance improvement
- ✅ How to read and modify RCCL source code
- ✅ How to tune RCCL for your specific workload
- ✅ How to debug common issues
- ✅ How to measure and validate improvements

---

## 📊 Expected Performance Gains

Following the optimization roadmap:
- **Week 2:** +10-20% (environment tuning)
- **Week 5:** +20-40% (algorithm optimization)
- **Week 8:** +40-60% (kernel optimization)
- **Week 12:** +60-100% (advanced optimization)
- **Target:** **2x performance improvement**

---

## 🔗 Important Links

**Documentation:**
- This directory: `/Users/ahalperin/xai/amd-dev/tools/docs/rccl/`

**Code:**
- RCCL source: `/Users/ahalperin/xai/amd-dev/amd/rccl/`
- RCCL tests: `/Users/ahalperin/xai/amd-dev/amd/rccl-tests/`

**Key Files:**
- Algorithm selection: `rccl/src/graph/tuning.cc`
- Topology detection: `rccl/src/graph/topo.cc`
- GPU primitives: `rccl/src/device/primitives.h`

---

## ❓ Common Questions

**Q: Where do I start if I know nothing about RCCL?**
- A: Read [rccl-design-overview.md](rccl-design-overview.md) first

**Q: My performance is low, what do I do?**
- A: See [rccl-bottleneck-analysis.md - Common Bottleneck Patterns](rccl-bottleneck-analysis.md#common-bottleneck-patterns)

**Q: I need to modify RCCL code, where do I look?**
- A: See [rccl-technical-internals.md](rccl-technical-internals.md) for detailed code analysis

**Q: What environment variables should I tune?**
- A: See [quick-reference.md - Tuning Environment Variables](quick-reference.md#tuning-environment-variables)

**Q: How do I run tests?**
- A: See [quick-reference.md - Quick Start Commands](quick-reference.md#quick-start-commands)

---

## ✅ Documentation Quality

- **Comprehensive:** 6,000+ lines covering all aspects
- **Practical:** 50+ ready-to-use commands and scripts
- **Detailed:** 100+ source code references
- **Structured:** Clear reading paths for all skill levels
- **Actionable:** 20+ specific optimization strategies

---

## 📞 Need Help?

1. Check the [README.md FAQ](README.md#faq)
2. Review [quick-reference.md](quick-reference.md) for common issues
3. Follow the [systematic investigation workflow](rccl-bottleneck-analysis.md#systematic-investigation-workflow)
4. Refer to specific document sections based on your issue

---

## 🎯 Success Metrics

After using this documentation, you should be able to:
- ✅ Explain RCCL architecture to others
- ✅ Run and interpret rccl-tests output
- ✅ Profile and identify performance bottlenecks
- ✅ Implement optimizations safely
- ✅ Validate improvements with benchmarks
- ✅ Navigate RCCL source code confidently

---

## 🚀 Ready to Begin?

**Choose your starting point:**

- **New to RCCL?** → [rccl-design-overview.md](rccl-design-overview.md)
- **Need to optimize?** → [rccl-bottleneck-analysis.md](rccl-bottleneck-analysis.md)
- **Want a plan?** → [optimization-roadmap.md](optimization-roadmap.md)
- **Quick answers?** → [quick-reference.md](quick-reference.md)
- **Browse all?** → [INDEX.md](INDEX.md)

---

**Good luck with your RCCL optimization work!** 🎉

*Documentation created: October 30, 2025*
