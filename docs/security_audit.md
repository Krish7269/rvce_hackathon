# Security Audit Summary

## Threat Model
- Arbitrary Python produced by LLM.
- Potential file system tampering, network exfiltration, resource exhaustion.

## Mitigations
1. **Static Validation**
   - `CodeValidator` / `RestrictedPython` block dangerous tokens (`subprocess`, `os.remove`, `requests`, etc.).
   - AST parse ensures syntactic sanity before execution.
2. **Sandbox Execution**
   - Temporary working directory deleted post-run.
   - `python -I` with sanitized environment removes user site packages.
   - Optional `resource` limits cap address/data space on Unix.
3. **Filesystem Controls**
   - Scripts instructed to touch only `chart.png` + `analysis_output.csv`.
   - Executor compares paths within sandbox.
4. **Network Isolation**
   - No network-friendly modules allowed; env lacks credentials.
5. **Observation & Logging**
   - Structured `ExecutionResult` captures stdout/stderr/error for audit.

## Residual Risks
- Windows lacks RLIMIT enforcement; document requirement to run executor on Linux for strongest isolation.
- Third-party library imports (pandas, matplotlib, seaborn) assumed trusted; vendoring recommended for production.


