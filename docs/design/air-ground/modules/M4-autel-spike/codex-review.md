# M4 记录

四态 PASS/FAIL/SKIP/NOT_RUN + simulated|hardware；dry_run RTK 为 SKIP。  
`or True` 已删；`scripts/mission/g2_autel_spike.py` 机读 exit_code。  
测审 FIX：`require_hardware` 须关键项均为 `mode=hardware` PASS，且 `device_id` 非占位 `autel`。
