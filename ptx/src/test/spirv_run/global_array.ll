target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7"
target triple = "amdgcn-amd-amdhsa"

@asdas = protected addrspace(1) externally_initialized global [4 x [2 x i32]] [[2 x i32] [i32 -1, i32 2], [2 x i32] [i32 3, i32 0], [2 x i32] zeroinitializer, [2 x i32] zeroinitializer]
@foobar = protected addrspace(1) externally_initialized global [4 x [2 x i64]] [[2 x i64] [i64 -1, i64 2], [2 x i64] [i64 3, i64 0], [2 x i64] [i64 ptrtoint (ptr addrspace(1) @asdas to i64), i64 0], [2 x i64] zeroinitializer]

define protected amdgpu_kernel void @global_array(ptr addrspace(4) byref(i64) %"16", ptr addrspace(4) byref(i64) %"17") #0 {
"21":
  %"9" = alloca i1, align 1, addrspace(5)
  store i1 false, ptr addrspace(5) %"9", align 1
  %"6" = alloca i64, align 8, addrspace(5)
  %"7" = alloca i64, align 8, addrspace(5)
  %"8" = alloca i32, align 4, addrspace(5)
  %0 = alloca i64, align 8, addrspace(5)
  store i64 ptrtoint (ptr addrspace(1) @foobar to i64), ptr addrspace(5) %0, align 8
  %"10" = load i64, ptr addrspace(5) %0, align 8
  store i64 %"10", ptr addrspace(5) %"6", align 8
  %"11" = load i64, ptr addrspace(4) %"17", align 8
  store i64 %"11", ptr addrspace(5) %"7", align 8
  %"13" = load i64, ptr addrspace(5) %"6", align 8
  %"19" = inttoptr i64 %"13" to ptr addrspace(1)
  %"12" = load i32, ptr addrspace(1) %"19", align 4
  store i32 %"12", ptr addrspace(5) %"8", align 4
  %"14" = load i64, ptr addrspace(5) %"7", align 8
  %"15" = load i32, ptr addrspace(5) %"8", align 4
  %"20" = inttoptr i64 %"14" to ptr addrspace(1)
  store i32 %"15", ptr addrspace(1) %"20", align 4
  ret void
}

attributes #0 = { "amdgpu-unsafe-fp-atomics"="true" "denormal-fp-math"="ieee,ieee" "denormal-fp-math-f32"="ieee,ieee" "no-trapping-math"="true" "uniform-work-group-size"="true" }
