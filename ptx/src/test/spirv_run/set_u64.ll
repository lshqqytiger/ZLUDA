target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7"
target triple = "amdgcn-amd-amdhsa"

define protected amdgpu_kernel void @set_u64(ptr addrspace(4) byref(i64) %"31", ptr addrspace(4) byref(i64) %"32") #0 {
  %"10" = alloca i1, align 1, addrspace(5)
  %"4" = alloca i64, align 8, addrspace(5)
  %"5" = alloca i64, align 8, addrspace(5)
  %"6" = alloca i64, align 8, addrspace(5)
  %"7" = alloca i64, align 8, addrspace(5)
  %"8" = alloca i32, align 4, addrspace(5)
  %"9" = alloca i32, align 4, addrspace(5)
  br label %1

1:                                                ; preds = %0
  store i1 false, ptr addrspace(5) %"10", align 1
  %"11" = load i64, ptr addrspace(4) %"31", align 8
  store i64 %"11", ptr addrspace(5) %"4", align 8
  %"12" = load i64, ptr addrspace(4) %"32", align 8
  store i64 %"12", ptr addrspace(5) %"5", align 8
  %"14" = load i64, ptr addrspace(5) %"4", align 8
  %"34" = inttoptr i64 %"14" to ptr
  %"33" = load i64, ptr %"34", align 8
  store i64 %"33", ptr addrspace(5) %"6", align 8
  %"16" = load i64, ptr addrspace(5) %"4", align 8
  %"35" = inttoptr i64 %"16" to ptr
  %"46" = getelementptr inbounds i8, ptr %"35", i64 8
  %"36" = load i64, ptr %"46", align 8
  store i64 %"36", ptr addrspace(5) %"7", align 8
  %"18" = load i64, ptr addrspace(5) %"6", align 8
  %"19" = load i64, ptr addrspace(5) %"7", align 8
  %2 = icmp ugt i64 %"18", %"19"
  %"37" = zext i1 %2 to i32
  store i32 %"37", ptr addrspace(5) %"8", align 4
  %"21" = load i64, ptr addrspace(5) %"7", align 8
  %"22" = load i64, ptr addrspace(5) %"6", align 8
  %3 = icmp eq i64 %"21", %"22"
  %"40" = zext i1 %3 to i32
  store i32 %"40", ptr addrspace(5) %"9", align 4
  %"23" = load i64, ptr addrspace(5) %"5", align 8
  %"24" = load i32, ptr addrspace(5) %"8", align 4
  %"43" = inttoptr i64 %"23" to ptr
  store i32 %"24", ptr %"43", align 4
  %"25" = load i64, ptr addrspace(5) %"5", align 8
  %"26" = load i32, ptr addrspace(5) %"9", align 4
  %"44" = inttoptr i64 %"25" to ptr
  %"48" = getelementptr inbounds i8, ptr %"44", i64 4
  store i32 %"26", ptr %"48", align 4
  ret void
}

attributes #0 = { "amdgpu-unsafe-fp-atomics"="true" "denormal-fp-math"="ieee,ieee" "denormal-fp-math-f32"="ieee,ieee" "no-trapping-math"="true" "uniform-work-group-size"="true" }
