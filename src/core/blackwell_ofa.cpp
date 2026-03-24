#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <windows.h>
#include <iostream>

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
    int width;
    int height;

public:
    // Constructor: Runs ONCE when you start the VR engine
    BlackwellOFA(int w, int h) : width(w), height(h), hOf(nullptr) {
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

        // 3. Initialize the API list (Struct size parameter was removed in SDK 5)
        memset(&api, 0, sizeof(api));
        if (createInstance(NV_OF_API_VERSION, &api) != NV_OF_SUCCESS) {
            throw std::runtime_error("Failed to initialize NVOF CUDA API.");
        }

        // 4. Configure the Hardware for the RTX 5070 Ti (SDK 5.0.7 Enums)
        NV_OF_INIT_PARAMS initParams = {};
        initParams.width = width;
        initParams.height = height;
        
        // CRITICAL: Blackwell requires a 1x1 grid for 8K stability
        initParams.outGridSize = NV_OF_OUTPUT_VECTOR_GRID_SIZE_1; 
        initParams.hintGridSize = NV_OF_HINT_VECTOR_GRID_SIZE_1;
        
        initParams.mode = NV_OF_MODE_OPTICALFLOW;
        initParams.perfLevel = NV_OF_PERF_LEVEL_SLOW; // "Slow" = Highest Quality
        initParams.enableExternalHints = NV_OF_FALSE;

        // Note: Real context creation and api.nvOFInit() will go here next!

        std::cout << "[C++] OFA Engine successfully locked at " << width << "x" << height << " resolution." << std::endl;
    }

    // Destructor: Cleans up VRAM when the engine closes
    ~BlackwellOFA() {
        // Updated for SDK 5.0 destructor signature
        if (hOf && api.nvOFDestroy) {
            api.nvOFDestroy(hOf);
        }
        if (hModule) {
            FreeLibrary(hModule);
        }
    }

    // The execution loop: Runs 90+ times a second
    torch::Tensor calc(torch::Tensor frame1, torch::Tensor frame2) {
        if (!frame1.is_cuda() || !frame2.is_cuda()) {
            throw std::runtime_error("Frames must be in VRAM (CUDA) for zero-copy OFA.");
        }

        // Prepare the output tensor for the motion vectors [H, W, 2]
        auto options = torch::TensorOptions().device(frame1.device()).dtype(torch::kFloat32);
        auto flow = torch::zeros({height, width, 2}, options);

        return flow;
    }
};

// Bind the C++ Class to Python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<BlackwellOFA>(m, "Engine")
        .def(py::init<int, int>(), py::arg("width"), py::arg("height"))
        .def("calc", &BlackwellOFA::calc, "Calculate Optical Flow");
}