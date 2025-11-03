# Macro Indexing in RCCL with clangd

## ⚠️ CONFIRMED LIMITATION: Functions as Macro Arguments

**clangd has a fundamental limitation with functions passed AS ARGUMENTS to macros.**

### The Problem:
```c
NCCLCHECKGOTO(taskAppend(info->comm, info), ret, fail);
              ^^^^^^^^^^
              This reference is NOT indexed by clangd ❌
```

### Why This Happens:
1. `taskAppend` is passed as an **argument** to the `NCCLCHECKGOTO` macro
2. clangd's AST parser processes macro **arguments** before expansion
3. At argument parsing time, `taskAppend(...)` is treated as unparsed tokens
4. Only after substitution into the macro body does it become a function call
5. By then, clangd has lost the connection to the original argument location

This is a **known architectural limitation** of clangd, not a configuration issue.

### Workarounds:

#### Option 1: Use grep/ripgrep (RECOMMENDED) ✅
```bash
# Find all references to taskAppend
grep -rn "\btaskAppend\b" /path/to/rccl/src/

# In VSCode:
# 1. Press Cmd+Shift+F (Search in Files)
# 2. Type: \btaskAppend\b
# 3. Enable regex mode (.*)
# 4. Set scope: amd-dev/amd/rccl/src/**/*.cc
```

**Result for taskAppend:**
```
2576:static ncclResult_t taskAppend(struct ncclComm* comm, struct ncclInfo* info) {
2745:  NCCLCHECKGOTO(taskAppend(info->comm, info), ret, fail);
```
✅ **This always works, including inside macros!**

#### Option 2: Use Microsoft C/C++ Extension
The Microsoft C/C++ extension uses a different indexer that DOES track macro arguments.
However, it conflicts with clangd and is slower. To try:
1. Disable clangd extension
2. Enable Microsoft C/C++ extension  
3. Wait for IntelliSense to index
4. Try "Find All References"

#### Option 3: Manual In-File Search
Many functions in RCCL are `static`, so they're only used within one file.
Use VSCode's in-file search: `Cmd+F`

---

## The Challenge

RCCL uses extensive macros for error handling (NCCLCHECK, NCCLCHECKGOTO, etc.). Function calls inside these macros need to be properly indexed for "Find All References" to work.

## Example

```c
// Macro definition in checks.h
#define NCCLCHECKGOTO(call, RES, label) do { \
  RES = call; \
  if (RES != ncclSuccess && RES != ncclInProgress) { \
    goto label; \
  } \
} while (0)

// Usage in enqueue.cc
NCCLCHECKGOTO(taskAppend(info->comm, info), ret, fail);
//            ^^^^^^^^^^
//            This function call should be indexed!
```

## How clangd Handles Macros

clangd performs **semantic analysis** on code, which includes:

1. **Preprocessing**: Expands all macros
2. **Parsing**: Builds an AST (Abstract Syntax Tree) from the expanded code
3. **Indexing**: Records all symbols and their references

So `NCCLCHECKGOTO(taskAppend(...))` becomes `ret = taskAppend(...)` during preprocessing, and clangd indexes the `taskAppend` call.

## Configuration for Better Macro Support

Our `.clangd` file includes:

```yaml
CompileFlags:
  Add:
    # Enable detailed macro expansion tracking
    - "-fmacro-backtrace-limit=0"
```

This ensures clangd tracks the full macro expansion chain for better diagnostics and cross-references.

## Testing Macro Indexing

### Test Case 1: NCCLCHECKGOTO with taskAppend

**File**: `src/enqueue.cc`

**Definition** (line 2576):
```c
static ncclResult_t taskAppend(struct ncclComm* comm, struct ncclInfo* info) {
  // ...
}
```

**Usage in macro** (line 2745):
```c
NCCLCHECKGOTO(taskAppend(info->comm, info), ret, fail);
```

**How to test**:
1. Open `src/enqueue.cc`
2. Go to line 2576
3. Click on "taskAppend" in the function definition
4. Press `Shift+F12` (Find All References)

**Expected result**: 2 references
- Line 2576: Definition
- Line 2745: Usage inside macro

### Test Case 2: NCCLCHECK with sendConnect

**File**: `src/transport/net.cc`

**Definition** (line 351):
```c
static ncclResult_t sendConnect(struct ncclComm* comm, ...) {
  // ...
}
```

**Usage in struct** (line 2054):
```c
struct ncclTransport netTransport = {
  "NET",
  canConnect,
  { sendSetup, sendConnect, sendFree, ... },
  //           ^^^^^^^^^^^
  { recvSetup, recvConnect, ... }
};
```

**How to test**:
1. Open `src/transport/net.cc`
2. Go to line 351
3. Click on "sendConnect"
4. Press `Shift+F12`

**Expected result**: 2 references
- Line 351: Definition
- Line 2054: Usage as function pointer

## Troubleshooting

### Issue: "Find All References" only shows definition, not usage in macro

**Diagnosis**: clangd hasn't fully indexed the file or needs to be restarted.

**Solutions**:

1. **Restart clangd** (most common fix):
   ```
   Cmd+Shift+P → "clangd: Restart language server"
   ```
   Wait for indexing to complete (watch status bar).

2. **Clear clangd cache**:
   ```bash
   rm -rf ~/.cache/clangd/
   rm -rf /Users/ahalperin/xai/amd-dev/amd/rccl/.cache/clangd/
   ```
   Then restart clangd.

3. **Verify compile_commands.json is up to date**:
   ```bash
   cd /Users/ahalperin/xai/amd-dev/amd/rccl
   stat compile_commands.json  # Check timestamp
   ```
   If outdated, regenerate:
   ```bash
   cd /Users/ahalperin/xai/amd-dev/tools/indexing/rccl
   ./setup.sh
   ```

4. **Check clangd is active**:
   - Open a `.cc` file in VSCode
   - Look at bottom status bar
   - Should say "clangd: idle" or "clangd: indexing"
   - If it says "C/C++", clangd is NOT active!

5. **Check for extension conflicts**:
   - Open Extensions (Cmd+Shift+X)
   - Search for "C/C++"
   - Disable Microsoft C/C++ extension for this workspace
   - Keep only clangd extension enabled

### Issue: clangd indexing is slow

**Cause**: Large codebase (263 files including headers).

**Solutions**:
- Be patient! Initial indexing can take 2-5 minutes
- Watch the status bar for progress
- Don't interrupt the indexing process
- Subsequent reopens will use cached index (much faster)

### Issue: Some macros work, others don't

**Cause**: Complex nested macros or macros with stringification (#) or token pasting (##) can sometimes confuse static analysis.

**Solutions**:
- For critical macros, expand them manually in critical paths
- Use inline functions instead of macros where possible
- Check that all header files are indexed (should be 167 .h files)

## Verification

Run the verification script to check indexing status:

```bash
cd /Users/ahalperin/xai/amd-dev/tools/indexing/rccl
./verify.sh
```

This will show:
- Number of indexed files
- Cache size and location
- Whether specific files are indexed
- Timestamp of last index update

## Key Points

1. ✅ clangd DOES support macro expansion for indexing
2. ✅ Function calls inside macros ARE indexed
3. ⚠️  You MUST restart clangd after configuration changes
4. ⚠️  Initial indexing takes time; be patient
5. ⚠️  Clear cache if you have persistent issues

## Common RCCL Macros

All of these should work with "Find All References":

- `NCCLCHECK(call)` - Basic error check
- `NCCLCHECKGOTO(call, RES, label)` - Error check with goto
- `CUDACHECK(cmd)` - CUDA error check
- `CUDACHECKGOTO(cmd, RES, label)` - CUDA error check with goto
- `NCCLCHECKTHREAD(a, args)` - Thread-safe check

All function calls inside these macros will be indexed if:
1. `compile_commands.json` includes the file
2. clangd has indexed the file
3. clangd language server is running

