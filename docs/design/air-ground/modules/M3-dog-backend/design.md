# M3 · DogSdk 显式 mode + FakeNav

`DogSdkAdapter(mode="stub"|"backend")`：backend 缺任一 backend → 立即报错；stub 明确用 DogStub。  
FakeNav 可观测 goto/cancel；D01–D04 单测。不声称真机 G1。
