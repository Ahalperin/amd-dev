# RCCL Branch Analysis: develop vs drop/2025-08

**Analysis Date:** October 28, 2025  
**Repository:** `/Users/ahalperin/xai/amd-dev/amd/rccl/`

## Executive Summary

The `develop` branch is **107 commits ahead** of the `drop/2025-08` branch. The `drop/2025-08` branch was last updated on **August 15, 2025**, while the `develop` branch is current as of **October 28, 2025** - representing approximately **2.5 months** of active development.

**Key Finding:** All changes are in `develop` branch only - there are **0 commits** in `drop/2025-08` that are not in `develop`, indicating a clean forward progression.

## Change Statistics

- **Total Commits Ahead:** 107 commits
- **Files Changed:** 202 files
- **Lines Added:** +18,411
- **Lines Removed:** -3,289
- **Net Change:** +15,122 lines

## Change Categories Breakdown

### 1. **Testing & Code Coverage** (~18 commits)
Significant expansion of unit tests and code coverage:
- Added tests for `rccl_wrap` (multiple PRs: #1890, #1895, #1855)
- Added tests for `Ipcsocket` (#1690)
- Added tests for `transport/p2p.cc` (#1774)
- Added tests for `transport/shm.cc` (#1689)
- Added tests for `coll_reg` (#1700)
- Added tests for `graph/xml.cc` & `graph/xml.h` (#1833)
- Added tests for `param.cc` (#1872)
- Added tests for `net_socket.cc` (#1840)
- Added tests for `proxy.cc` (#1818)
- Added tests for `enqueue.cc` (#1853)
- Added tests for `comm.h` (#1783)
- Added unit test for `bitops.h` (#1821)
- Added tests for `allocator.h` (#1676)

### 2. **Bug Fixes & Stability** (~20 commits)
Critical stability and correctness fixes:
- **MSCCL data corruption fix** (#1960)
- **Segfault fixes:**
  - ext-profiler plugin (#1986)
  - profiler plugin (#1973)
  - libibverbs 0 device (#1820)
- **LL Protocol hang fix for gfx950** (#1932)
- MSCCL++ null deref fix (#1959)
- Fixed `ar_with_bias` test issue (#1912)
- Fixed `TestBedChild` hang (#1875)
- Fixed memory leak in tests (#1863)
- Fixed LL128 proto selection (#1822)
- Fixed git version fetching logic (#1981)
- Fixed build failure in rccl_prim_test (#1984)

### 3. **GPU Architecture Support** (~7 commits)
Enhancements for AMD GPU architectures:

#### **gfx950 (MI350X) Support:**
- Enabling gdrcopy option (#1955)
- Make bypassing `__threadfence` default for multinode (#1947)
- Channel tuning for ReduceScatter and AllGather (#1940)
- LL Protocol missing fences fix (#1932)
- Support in topo_expl tool (#1829)
- Dynamic fetch/reduce pipelining (#1861)

#### **gfx942 Support:**
- Enable LL128 and tuning table for 4 NICs (#1898)

#### **gfx940 Support:**
- Add reduce/broadcast algo/proto selection table for multi-node (#1889)

### 4. **Performance Optimizations** (~4 commits)
- **Batched P2P** for enhanced alltoall small message performance (#1902)
- **Direct allgather algorithm** (#1868)
- **Device threadfence optimization** for LL64 protocol (#1858)
- **Dynamic fetch/reduce pipelining** for reduction collectives - Simple protocol (#1861)

### 5. **NCCL Sync Updates** (~4 commits)
Updates from upstream NCCL:
- **NCCL 2.27.7-1** sync (#1928)
- **NCCL 2.27.3-1** sync (#1880, #1892)
- Updated `RCCL_API_TRACE_VERSION_PATCH` to 2 (#1916)
- `ncclDevFuncId` 64-bit keyed map with field packing (#1857)

### 6. **CI/Build Infrastructure** (~15 commits)
Substantial CI and build improvements:
- **TheRock CI integration:**
  - Enable Presubmit CI Gating (#1954)
  - Add single node tests (#1876)
  - Add CI badge (#1874, #1851)
- **Azure CI:** Switch to ROCm 6.4.1 and add rccl-tests (#1782)
- **External CI:** Add references to rocm-systems super repo (#1935)
- Enable ccache w/ namespace for external use (#1966)
- Stop generating sym kernels by default (#1907)
- Disable MSCCL++ compilation by default (#1879)
- Fix UT packaging on Debian OS (#1848, #1854, #1846, #1831)
- Add .clang-format for C++ code (#1404)

### 7. **Profiling & Debugging** (~10 commits)
Enhanced debugging and profiling capabilities:
- **Collective latency profiler** (#1785)
- Fix ext-profiler plugin segfault (#1986)
- Fix profiler plugin segfault (#1973)
- Add roctracer and rocm-core include directories (#1970)
- rocprofiler-sdk codeowner updates (#1974, #1933)
- Update npkit_trace_generator.py (#1891)
- Dump compiler-determined GPU kernel resource usage (#1965)
- Enable more events for LL128 NPKIT trace collection (#1827)
- Device allocation tracker (#1878)

### 8. **MSCCL Changes** (~4 commits)
MSCCL-related updates:
- **MSCCL data corruption fix** (#1960)
- MSCCL++ fix split path null deref (#1959)
- Disable msccl for fp8 datatype (#1888)
- Disable MSCCL++ compilation by default (#1879)

### 9. **API & Feature Additions**
- **Fused all reduce and elementwise operations** (#1729)
- Upcast FP8 to Half (FP16) for Sum Operation (#1775)
- Add optional bf16 software-triggered pipelining for reduceCopyPacks (#1758)
- Expose symbols for RCCL algo/proto/channels selection functions (#1923)
- Force enable proto/algo after model selection (#1799)
- Passing down NET_OPTIONAL_RECV_COMPLETION hint to network plugin (#1752)
- Add support for additional paths in RCCL DMABUF kernel configuration loading (#1825)

### 10. **Memory Management**
- Hugepages backed host buffer for larger allocations (#1841)
  - Note: This was reverted (#1951) then re-applied
- Device allocation tracker (#1878)
- Disable graph mode memory registration and UBR as unsupported (#1977)

### 11. **Documentation & Maintenance**
- Add environment variables reference page (no PR)
- Update help text in README (#1837)
- Update RCCL Replayer README.md (#1870)
- Fix Docker guide formatting (#1882)
- Add reference to supported data types section (#1893)
- Add usage tip for ignore cpu affinity (#1948)
- Updated CODEOWNERS (#2010, #1921, #1917, #1915, #1869)
- Add MIT license file (#1908)
- Bump minimum cmake version to 3.16 (#1909)

## Key Files Modified

### Core Source Files (src/):
- `src/device/prims_simple.h` - Simple protocol primitives
- `src/device/prims_ll.h` - LL protocol primitives
- `src/device/prims_ll128.h` - LL128 protocol primitives
- `src/device/common.h`, `common_kernel.h` - Common device code
- `src/device/all_reduce.h`, `all_gather.h`, `reduce.h` - Collective operations
- `src/collectives.cc` - Collective implementations
- `src/channel.cc` - Channel management
- `src/bootstrap.cc` - Bootstrap logic
- `src/allocator.cc` - Memory allocation

### Test Files:
- 13 test files modified/added
- Major additions to `test/RcclWrapTests.cpp` (+2,378 lines!)
- New: `test/ProxyTests.cpp` (+433 lines)
- Updates to `test/ParamTests.cpp`, `test/common/TestBedChild.cpp`

### Tools:
- 11 tool files modified
- `tools/topo_expl/` - Support for gfx950, FMT dependency resolution
- `tools/RcclReplayer/README.md` - Updated documentation
- `tools/scripts/npkit_trace_generator.py` - NPKit improvements
- `tools/JitterBench/runSweep.sh` - Benchmark fixes

## Critical Changes for Production Use

### **High Priority - Stability:**
1. ✅ MSCCL data corruption fix (#1960)
2. ✅ Multiple segfault fixes (profiler, libibverbs)
3. ✅ gfx950 LL Protocol hang fix (#1932)
4. ✅ MSCCL++ null deref fix (#1959)

### **High Priority - Performance:**
1. ✅ gfx950 channel tuning and optimization
2. ✅ Batched P2P for alltoall performance
3. ✅ Direct allgather algorithm
4. ✅ threadfence optimizations

### **Medium Priority - Features:**
1. ✅ Fused operations support
2. ✅ FP8/FP16 support enhancements
3. ✅ Latency profiler
4. ✅ Enhanced debugging capabilities

### **Medium Priority - Infrastructure:**
1. ✅ TheRock CI integration
2. ✅ ROCm 6.4.1 support
3. ✅ Build system improvements

## Risk Assessment

### **Low Risk:**
- Test additions (only improve coverage)
- Documentation updates
- CI/CD improvements
- New optional features with flags

### **Medium Risk:**
- MSCCL changes (disabled by default in some cases)
- Hugepages implementation (was reverted once)
- Build system changes (may affect compilation)

### **Requires Validation:**
- gfx950-specific changes (if using MI350X)
- LL Protocol changes (verify multi-node stability)
- Performance optimizations (benchmark before/after)

## Recommendations

### **For Production Deployment:**
1. **Upgrade Recommended:** The `develop` branch contains critical bug fixes (data corruption, segfaults, hangs)
2. **Testing Priority:**
   - Multi-node LL protocol functionality
   - MSCCL operations (if used)
   - Profiler plugin integration
   - AllReduce with bias operations

### **For MI350X (gfx950) Users:**
- The `develop` branch has **essential fixes** for gfx950:
  - LL Protocol hang fix (#1932)
  - Memory fence corrections
  - Performance tuning

### **For CI/CD Integration:**
- Consider adopting TheRock CI setup for automated testing
- ccache namespace support available for external builds

### **Next Steps:**
1. Review specific PRs relevant to your workload
2. Test develop branch in staging environment
3. Validate performance benchmarks
4. Check compatibility with your network plugin version

## Version Information

- **drop/2025-08 latest commit:** `c3b8de4e` (2025-08-15)
  - "[DEVICE] Use noinline for LLGenericOp only on gfx950 (#1849)"
  
- **develop latest commit:** `f290e302` (2025-10-28)
  - "Updated CODEOWNERS to instead use RCCL-Reviewers team (#2010)"

## Upstream NCCL Versions Included

- NCCL 2.27.3-1
- NCCL 2.27.5-1
- NCCL 2.27.6-1
- NCCL 2.27.7-1

---

**Analysis Tool:** Git diff and log analysis  
**Generated by:** Automated branch comparison script  
**Note:** This analysis is based on commit messages and git statistics. Detailed code review recommended for production deployment decisions.

