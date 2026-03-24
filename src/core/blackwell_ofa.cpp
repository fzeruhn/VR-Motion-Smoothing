#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <windows.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>

// Include the NVIDIA Optical Flow CUDA Headers
#include "nvOpticalFlowCuda.h"

namespace py = pybind11;

// Define the function signature for dynamic DLL loading
typedef NV_OF_STATUS(NVOFAPI* PFNNvOFAPICreateInstanceCuda)(uint32_t nvOfVersion, NV_OF_CUDA_API_FUNCTION_LIST* pCudaRet);

class BlackwellOFA {
private:
    HMODULE hModule;
    NV_OF_CUDA_API_FUNCTION_LIST api;
    NvOFHandle hOf;
    
    // Hardware-Managed Memory Buffers
    NvOFGPUBufferHandle hIn1;
    NvOFGPUBufferHandle hIn2;
    NvOFGPUBufferHandle hOut;
    
    int width;
    int height;

public:
    // Constructor: Runs ONCE when you start the VR engine
    BlackwellOFA(int w, int h) : width(w), height(h), hOf(nullptr), hIn1(nullptr), hIn2(nullptr), hOut(nullptr) {
        std::cout << "[C++] Waking up Blackwell OFA Silicon..." << std::endl;

        // 1. Dynamically load the NVIDIA Driver DLL
        hModule = LoadLibraryA("nvOfAPI64.dll");
        if (!hModule) {
            throw std::runtime_error("Failed to load nvOfAPI64.dll. Ensure NVIDIA drivers are up to date.");
        }

        // 2. Map the CUDA Instance function
        auto createInstance = (PFNNvOFAPICreateInstanceCuda)GetProcAddress(hModule, "NvOFAPICreateInstanceCuda");
        if (!createInstance) {
            throw std::runtime_error("Failed to map NvOFAPICreateInstanceCuda from driver.");
        }

        // 3. Initialize the API list 
        memset(&api, 0, sizeof(api));
        if (createInstance(NV_OF_API_VERSION, &api) != NV_OF_SUCCESS) {
            throw std::runtime_error("Failed to initialize NVOF CUDA API.");
        }

        // 4. Bind to PyTorch's CUDA Context
        torch::Tensor dummy = torch::zeros({1}, torch::kCUDA);
        CUcontext cuContext;
        cuCtxGetCurrent(&cuContext);
        if (!cuContext) {
            throw std::runtime_error("Failed to find PyTorch CUDA context.");
        }

        // 5. Create the OFA Instance attached to the GPU
        if (api.nvCreateOpticalFlowCuda(cuContext, &hOf) != NV_OF_SUCCESS) {
            throw std::runtime_error("Failed to bind OFA instance to CUDA context.");
        }

        // 6. Configure the Hardware for the RTX 5070 Ti
        NV_OF_INIT_PARAMS initParams = {};
        initParams.width = width;
        initParams.height = height;
        initParams.outGridSize = NV_OF_OUTPUT_VECTOR_GRID_SIZE_1; 
        initParams.hintGridSize = NV_OF_HINT_VECTOR_GRID_SIZE_1;
        initParams.mode = NV_OF_MODE_OPTICALFLOW;
        initParams.perfLevel = NV_OF_PERF_LEVEL_SLOW; // Highest Quality
        initParams.enableExternalHints = NV_OF_FALSE;

        if (api.nvOFInit(hOf, &initParams) != NV_OF_SUCCESS) {
            throw std::runtime_error("Hardware rejected 8K configuration parameters.");
        }

        // ==========================================
        // 7. ALLOCATE DEDICATED HARDWARE BUFFERS
        // ==========================================
        NV_OF_BUFFER_DESCRIPTOR inDesc = {};
        inDesc.width = width;
        inDesc.height = height;
        inDesc.bufferFormat = NV_OF_BUFFER_FORMAT_GRAYSCALE8; 
        inDesc.bufferUsage = NV_OF_BUFFER_USAGE_INPUT;

        // Using the strict CUDA API to allocate hardware boundaries
        if (api.nvOFCreateGPUBufferCuda(hOf, &inDesc, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, &hIn1) != NV_OF_SUCCESS) throw std::runtime_error("Failed to allocate Input Buffer 1");
        if (api.nvOFCreateGPUBufferCuda(hOf, &inDesc, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, &hIn2) != NV_OF_SUCCESS) throw std::runtime_error("Failed to allocate Input Buffer 2");

        NV_OF_BUFFER_DESCRIPTOR outDesc = {};
        outDesc.width = width;
        outDesc.height = height;
        outDesc.bufferFormat = NV_OF_BUFFER_FORMAT_SHORT2; 
        outDesc.bufferUsage = NV_OF_BUFFER_USAGE_OUTPUT;

        if (api.nvOFCreateGPUBufferCuda(hOf, &outDesc, NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, &hOut) != NV_OF_SUCCESS) throw std::runtime_error("Failed to allocate Output Buffer");

        std::cout << "[C++] OFA Engine & Memory Buffers successfully locked at " << width << "x" << height << " resolution." << std::endl;
    }

    // Destructor: Cleans up VRAM when the engine closes
    ~BlackwellOFA() {
        if (hIn1 && api.nvOFDestroyGPUBufferCuda) api.nvOFDestroyGPUBufferCuda(hIn1);
        if (hIn2 && api.nvOFDestroyGPUBufferCuda) api.nvOFDestroyGPUBufferCuda(hIn2);
        if (hOut && api.nvOFDestroyGPUBufferCuda) api.nvOFDestroyGPUBufferCuda(hOut);
        
        if (hOf && api.nvOFDestroy) {
            api.nvOFDestroy(hOf);
        }
        if (hModule) {
            FreeLibrary(hModule);
        }
    }

    // The execution loop: Runs 90+ times a second
    torch::Tensor calc(torch::Tensor frame1, torch::Tensor frame2) {
        if (!frame1.is_cuda() || !frame2.is_cuda()) throw std::runtime_error("Frames must be in VRAM (CUDA).");
        if (frame1.scalar_type() != torch::kUInt8 || frame2.scalar_type() != torch::kUInt8) {
            throw std::runtime_error("Frames must be 8-bit unsigned integers (torch.uint8).");
        }

        // Extract the raw raw VRAM pointers from the Hardware Handles
        CUdeviceptr ptr_in1 = api.nvOFGPUBufferGetCUdeviceptr(hIn1);
        CUdeviceptr ptr_in2 = api.nvOFGPUBufferGetCUdeviceptr(hIn2);
        CUdeviceptr ptr_out = api.nvOFGPUBufferGetCUdeviceptr(hOut);

        // 8. Safely copy PyTorch memory into NVIDIA Hardware Buffers
        cudaMemcpy((void*)ptr_in1, frame1.data_ptr(), width * height * sizeof(uint8_t), cudaMemcpyDeviceToDevice);
        cudaMemcpy((void*)ptr_in2, frame2.data_ptr(), width * height * sizeof(uint8_t), cudaMemcpyDeviceToDevice);

        // 9. Queue up the Execution
        NV_OF_EXECUTE_INPUT_PARAMS exeInParams = {};
        exeInParams.inputFrame = hIn1;
        exeInParams.referenceFrame = hIn2;
        exeInParams.disableTemporalHints = NV_OF_TRUE;

        NV_OF_EXECUTE_OUTPUT_PARAMS exeOutParams = {};
        exeOutParams.outputBuffer = hOut;

        // 10. FIRE THE SILICON
        NV_OF_STATUS status = api.nvOFExecute(hOf, &exeInParams, &exeOutParams);

        if (status != NV_OF_SUCCESS) {
            std::cerr << "[C++] WARNING: OFA Execution Failed with status code: " << status << std::endl;
        }

        // 11. Prepare PyTorch Output Tensor 
        auto options = torch::TensorOptions().device(frame1.device()).dtype(torch::kInt16);
        auto flow = torch::zeros({height, width, 2}, options);

        // 12. Copy the computed vectors back into PyTorch space
        cudaMemcpy(flow.data_ptr(), (void*)ptr_out, width * height * 2 * sizeof(int16_t), cudaMemcpyDeviceToDevice);

        return flow;
    }
};

// Bind the C++ Class to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<BlackwellOFA>(m, "Engine")
        .def(py::init<int, int>(), py::arg("width"), py::arg("height"))
        .def("calc", &BlackwellOFA::calc, "Calculate Optical Flow");
}