#include <iostream>
#include <onnxruntime_cxx_api.h>

int main()
{
    // ONNX Runtime 전체 환경 생성
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "gaze");

    // 모델 실행 옵션
    Ort::SessionOptions sessionOptions;

    // ONNX 모델 경로
    const wchar_t* modelPath =
        L"C:\\Users\\user\\Desktop\\lee\\Gaze-monitoring\\weights\\gaze_model.onnx";

    // 실제 ONNX 모델 로드
    Ort::Session session(env, modelPath, sessionOptions);

    std::cout << "ONNX model loaded successfully." << std::endl;
    // Ort::AllocatorWithDefaultOptions를 사용하여 메모리 할당자 생성
    Ort::AllocatorWithDefaultOptions allocator;

    //입력 개수 확인
    size_t inputCount = session.GetInputCount();
    std::cout << "Number of inputs: " << inputCount << std::endl;
    size_t outputCount = session.GetOutputCount();
    std::cout << "Number of outputs: " << outputCount << std::endl;

    auto inputName = session.GetInputNameAllocated(0, allocator);
    std::cout << "Input name: " << inputName.get() << std::endl;

    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();

    std::vector<int64_t> inputDims = inputTensorInfo.GetShape();
    std::cout << "input shape: ";

    for (int64_t dim : inputDims)
    {
        std::cout << dim << " ";
    }
    std::cout << std::endl;


    size_t inputTensorsize = 1 *3* 224 * 224; // 예시: 입력 텐서 크기 (배치 크기 * 채널 수 * 높이 * 너비)
    std::vector<float> inputTensorValues(inputTensorsize, 0.0f); // 예시: 입력 텐서 값 초기화

    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(), inputTensorsize, inputDims.data(), inputDims.size());

    auto outputName = session.GetOutputNameAllocated(0, allocator);
    std::cout << "Output name: " << outputName.get() << std::endl;

    const char* outputNames[] = {outputName.get()};
    const char* inputNames[] = {inputName.get()};

    // 모델 실행
    auto outputTensors = session.Run(Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1, outputNames, 1);

    float* outputData = outputTensors.front().GetTensorMutableData<float>();

    std::cout << "Output[0]" << outputData[0] << std::endl;
    std::cout << "Output[1]" << outputData[1] << std::endl;

    return 0;
}