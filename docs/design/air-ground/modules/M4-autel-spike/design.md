# M4 · Autel 四态 spike

状态机：`PASS|FAIL|SKIP|NOT_RUN` + `mode=simulated|hardware`。  
dry_run 只能 simulated；hardware PASS 须 device/时间戳。无 SDK 时真机项 NOT_RUN/SKIP，脚本可机读。删 `or True`。
