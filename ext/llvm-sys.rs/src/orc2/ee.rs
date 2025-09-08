use super::{super::execution_engine::*, *};

pub type LLVMMemoryManagerCreateContextCallback = extern "C" fn(CtxCtx: *mut ::libc::c_void);
pub type LLVMMemoryManagerNotifyTerminatingCallback = extern "C" fn(CtxCtx: *mut ::libc::c_void);

extern "C" {
    pub fn LLVMOrcCreateRTDyldObjectLinkingLayerWithSectionMemoryManager(
        ES: LLVMOrcExecutionSessionRef,
    ) -> LLVMOrcObjectLayerRef;
    pub fn LLVMOrcCreateRTDyldObjectLinkingLayerWithMCJITMemoryManagerLikeCallbacks(
        ES: LLVMOrcExecutionSessionRef,
        CreateContext: LLVMMemoryManagerCreateContextCallback,
        NotifyTerminating: LLVMMemoryManagerNotifyTerminatingCallback,
        AllocateCodeSection: LLVMMemoryManagerAllocateCodeSectionCallback,
        AllocateDataSection: LLVMMemoryManagerAllocateDataSectionCallback,
        FinalizeMemory: LLVMMemoryManagerFinalizeMemoryCallback,
        Destroy: LLVMMemoryManagerDestroyCallback,
    ) -> LLVMOrcObjectLayerRef;
    pub fn LLVMOrcRTDyldObjectLinkingLayerRegisterJITEventListener(
        RTDyldObjLinkingLayer: LLVMOrcObjectLayerRef,
        Listener: LLVMJITEventListenerRef,
    );
}
