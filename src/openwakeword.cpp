#include "openwakeword.hpp"
#include <SDL.h>
#include <SDL_audio.h>
#include <atomic>

// Constants for audio recording
const int SAMPLE_RATE = 16000; // Sample rate (Hz)
const int CHUNK_SIZE_MS = 80;  // Chunk size (milliseconds)
const int NUM_CHANNELS = 1;    // Number of audio channels (1 for mono)
const int SAMPLES_PER_CHUNK = SAMPLE_RATE * CHUNK_SIZE_MS / 1000; // Number of samples per chunk
const int BITS_PER_SAMPLE = 16; // Bits per sample (16-bit PCM)

const size_t chunkSamples = 1280; // 80 ms
const size_t numMels = 32;
const size_t embWindowSize = 76; // 775 ms
const size_t embStepSize = 8;    // 80 ms
const size_t embFeatures = 96;
const size_t wwFeatures = 16;

struct State
{
    Ort::Env env;
    std::vector<std::mutex> mutFeatures;
    std::vector<std::condition_variable> cvFeatures;
    std::vector<bool> featuresExhausted;
    std::vector<bool> featuresReady;
    size_t numReady;
    bool samplesExhausted = false, melsExhausted = false;
    bool samplesReady = false, melsReady = false;
    std::mutex mutSamples, mutMels, mutReady, mutOutput;
    std::condition_variable cvSamples, cvMels, cvReady;
    State()
        : mutFeatures(1), cvFeatures(1),
          featuresExhausted(1), featuresReady(1),
          numReady(0), samplesExhausted(false), melsExhausted(false),
          samplesReady(false), melsReady(false)
    {
        env = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_WARNING, "test");
        env.DisableTelemetryEvents();

        std::fill(featuresExhausted.begin(), featuresExhausted.end(), false);
        std::fill(featuresReady.begin(), featuresReady.end(), false);
    }
};

struct Settings
{
    std::filesystem::path melModelPath = std::filesystem::path("../model/melspectrogram.onnx");
    std::filesystem::path embModelPath = std::filesystem::path("../model/embedding_model.onnx");
    std::vector<std::filesystem::path> wwModelPaths;

    size_t frameSize = 4 * 1280; // 80 ms
    size_t stepFrames = 4;

    float threshold = 0.5f;
    int triggerLevel = 4;
    int refractory = 20;

    bool debug = false;

    Ort::SessionOptions options;
};

void audioToMels(Settings &settings, State &state, std::vector<float> &samplesIn,
                 std::vector<float> &melsOut)
{
    Ort::AllocatorWithDefaultOptions allocator;
    auto memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    auto melSession =
        Ort::Session(state.env, settings.melModelPath.c_str(), settings.options);

    std::vector<int64_t> samplesShape{1, (int64_t)settings.frameSize};

    auto melInputName = melSession.GetInputNameAllocated(0, allocator);
    std::vector<const char *> melInputNames{melInputName.get()};

    auto melOutputName = melSession.GetOutputNameAllocated(0, allocator);
    std::vector<const char *> melOutputNames{melOutputName.get()};

    std::vector<float> todoSamples;

    {
        std::unique_lock lockReady(state.mutReady);
        std::cerr << "[LOG] Loaded mel spectrogram model" << '\n';
        state.numReady += 1;
        state.cvReady.notify_one();
    }

    while (true)
    {
        {
            std::unique_lock lockSamples{state.mutSamples};
            state.cvSamples.wait(lockSamples,
                                 [&state]
                                 { return state.samplesReady; });
            if (state.samplesExhausted && samplesIn.empty())
            {
                break;
            }
            copy(samplesIn.begin(), samplesIn.end(), back_inserter(todoSamples));
            samplesIn.clear();

            if (!state.samplesExhausted)
            {
                state.samplesReady = false;
            }
        }

        while (todoSamples.size() >= settings.frameSize)
        {
            // Generate mels for audio samples
            std::vector<Ort::Value> melInputTensors;
            melInputTensors.push_back(Ort::Value::CreateTensor<float>(
                memoryInfo, todoSamples.data(), settings.frameSize,
                samplesShape.data(), samplesShape.size()));

            auto melOutputTensors =
                melSession.Run(Ort::RunOptions{nullptr}, melInputNames.data(),
                               melInputTensors.data(), melInputNames.size(),
                               melOutputNames.data(), melOutputNames.size());

            // (1, 1, frames, mels = 32)
            const auto &melOut = melOutputTensors.front();
            const auto melInfo = melOut.GetTensorTypeAndShapeInfo();
            const auto melShape = melInfo.GetShape();

            const float *melData = melOut.GetTensorData<float>();
            size_t melCount =
                accumulate(melShape.begin(), melShape.end(), 1, std::multiplies<>());

            {
                std::unique_lock lockMels{state.mutMels};
                for (size_t i = 0; i < melCount; i++)
                {
                    // Scale mels for Google speech embedding model
                    melsOut.push_back((melData[i] / 10.0f) + 2.0f);
                }
                state.melsReady = true;
                state.cvMels.notify_one();
            }

            todoSamples.erase(todoSamples.begin(),
                              todoSamples.begin() + settings.frameSize);
        }
    }

} // audioToMels

void melsToFeatures(Settings &settings, State &state, std::vector<float> &melsIn,
                    std::vector<std::vector<float>> &featuresOut)
{
    Ort::AllocatorWithDefaultOptions allocator;
    auto memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    auto embSession =
        Ort::Session(state.env, settings.embModelPath.c_str(), settings.options);

    std::vector<int64_t> embShape{1, (int64_t)embWindowSize, (int64_t)numMels, 1};

    auto embInputName = embSession.GetInputNameAllocated(0, allocator);
    std::vector<const char *> embInputNames{embInputName.get()};

    auto embOutputName = embSession.GetOutputNameAllocated(0, allocator);
    std::vector<const char *> embOutputNames{embOutputName.get()};

    std::vector<float> todoMels;
    size_t melFrames = 0;

    {
        std::unique_lock lockReady(state.mutReady);
        std::cerr << "[LOG] Loaded speech embedding model" << '\n';
        state.numReady += 1;
        state.cvReady.notify_one();
    }

    while (true)
    {
        {
            std::unique_lock lockMels{state.mutMels};
            state.cvMels.wait(lockMels, [&state]
                              { return state.melsReady; });
            if (state.melsExhausted && melsIn.empty())
            {
                break;
            }
            copy(melsIn.begin(), melsIn.end(), back_inserter(todoMels));
            melsIn.clear();

            if (!state.melsExhausted)
            {
                state.melsReady = false;
            }
        }

        melFrames = todoMels.size() / numMels;
        while (melFrames >= embWindowSize)
        {
            // Generate embeddings for mels
            std::vector<Ort::Value> embInputTensors;
            embInputTensors.push_back(Ort::Value::CreateTensor<float>(
                memoryInfo, todoMels.data(), embWindowSize * numMels, embShape.data(),
                embShape.size()));

            auto embOutputTensors =
                embSession.Run(Ort::RunOptions{nullptr}, embInputNames.data(),
                               embInputTensors.data(), embInputTensors.size(),
                               embOutputNames.data(), embOutputNames.size());

            const auto &embOut = embOutputTensors.front();
            const auto embOutInfo = embOut.GetTensorTypeAndShapeInfo();
            const auto embOutShape = embOutInfo.GetShape();

            const float *embOutData = embOut.GetTensorData<float>();
            size_t embOutCount =
                accumulate(embOutShape.begin(), embOutShape.end(), 1, std::multiplies<>());

            // Send to each wake word model
            for (size_t i = 0; i < featuresOut.size(); i++)
            {
                std::unique_lock lockFeatures{state.mutFeatures[i]};
                copy(embOutData, embOutData + embOutCount,
                     back_inserter(featuresOut[i]));
                state.featuresReady[i] = true;
                state.cvFeatures[i].notify_one();
            }

            // Erase a step's worth of mels
            todoMels.erase(todoMels.begin(),
                           todoMels.begin() + (embStepSize * numMels));

            melFrames = todoMels.size() / numMels;
        }
    }

} // melsToFeatures

void featuresToOutput(Settings &settings, State &state, size_t wwIdx,
                      std::vector<std::vector<float>> &featuresIn, std::atomic_bool &wake_word_detected)
{
    Ort::AllocatorWithDefaultOptions allocator;
    auto memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    auto wwModelPath = settings.wwModelPaths[wwIdx];
    auto wwName = wwModelPath.stem();
    auto wwSession =
        Ort::Session(state.env, wwModelPath.c_str(), settings.options);

    std::vector<int64_t> wwShape{1, (int64_t)wwFeatures, (int64_t)embFeatures};

    auto wwInputName = wwSession.GetInputNameAllocated(0, allocator);
    std::vector<const char *> wwInputNames{wwInputName.get()};

    auto wwOutputName = wwSession.GetOutputNameAllocated(0, allocator);
    std::vector<const char *> wwOutputNames{wwOutputName.get()};

    std::vector<float> todoFeatures;
    size_t numBufferedFeatures = 0;
    int activation = 0;

    {
        std::unique_lock lockReady(state.mutReady);
        std::cerr << "[LOG] Loaded " << wwName << " model" << '\n';
        state.numReady += 1;
        state.cvReady.notify_one();
    }

    while (true)
    {
        {
            std::unique_lock lockFeatures{state.mutFeatures[wwIdx]};
            state.cvFeatures[wwIdx].wait(
                lockFeatures, [&state, wwIdx]
                { return state.featuresReady[wwIdx]; });
            if (state.featuresExhausted[wwIdx] && featuresIn[wwIdx].empty())
            {
                break;
            }
            copy(featuresIn[wwIdx].begin(), featuresIn[wwIdx].end(),
                 back_inserter(todoFeatures));
            featuresIn[wwIdx].clear();

            if (!state.featuresExhausted[wwIdx])
            {
                state.featuresReady[wwIdx] = false;
            }
        }

        numBufferedFeatures = todoFeatures.size() / embFeatures;
        while (numBufferedFeatures >= wwFeatures)
        {
            std::vector<Ort::Value> wwInputTensors;
            wwInputTensors.push_back(Ort::Value::CreateTensor<float>(
                memoryInfo, todoFeatures.data(), wwFeatures * embFeatures,
                wwShape.data(), wwShape.size()));

            auto wwOutputTensors =
                wwSession.Run(Ort::RunOptions{nullptr}, wwInputNames.data(),
                              wwInputTensors.data(), 1, wwOutputNames.data(), 1);

            const auto &wwOut = wwOutputTensors.front();
            const auto wwOutInfo = wwOut.GetTensorTypeAndShapeInfo();
            const auto wwOutShape = wwOutInfo.GetShape();
            const float *wwOutData = wwOut.GetTensorData<float>();
            size_t wwOutCount =
                accumulate(wwOutShape.begin(), wwOutShape.end(), 1, std::multiplies<>());

            for (size_t i = 0; i < wwOutCount; i++)
            {
                auto probability = wwOutData[i];
                if (settings.debug)
                {
                    {
                        std::unique_lock lockOutput(state.mutOutput);
                        std::cerr << wwName << " " << probability << '\n';
                    }
                }

                if (probability > settings.threshold)
                {
                    // Activated
                    activation++;
                    if (activation >= settings.triggerLevel)
                    {
                        // Trigger level reached
                        {
                            std::unique_lock lockOutput(state.mutOutput);
                            std::cout << wwName << '\n';
                            wake_word_detected = true;
                        }
                        activation = -settings.refractory;
                    } 
                }
                else
                {
                    // Back towards 0
                    if (activation > 0)
                    {
                        activation = std::max(0, activation - 1);
                    }
                    else
                    {
                        activation = std::min(0, activation + 1);
                    }
                }
            }

            // Remove 1 embedding
            todoFeatures.erase(todoFeatures.begin(),
                               todoFeatures.begin() + (1 * embFeatures));

            numBufferedFeatures = todoFeatures.size() / embFeatures;
        }
    }

} // featuresToOutput

// Audio callback function
void audioCallback(void *userdata, uint8_t *stream, int len)
{
    std::vector<uint8_t> *audioBuffer = static_cast<std::vector<uint8_t> *>(userdata);
    audioBuffer->insert(audioBuffer->end(), stream, stream + len);
}

void run_thread(std::string path_to_model, std::atomic_bool &wake_word_detected, std::atomic_bool &do_exit)
{
    Settings _settings;
    _settings.wwModelPaths.push_back(std::filesystem::path(path_to_model));
    _settings.frameSize = _settings.stepFrames * chunkSamples;
    using namespace std;
    // Absolutely critical for performance
    _settings.options.SetIntraOpNumThreads(1);
    _settings.options.SetInterOpNumThreads(1);

    const size_t numWakeWords = _settings.wwModelPaths.size();
    State state;

    vector<float> floatSamples;
    vector<float> mels;
    vector<vector<float>> features(numWakeWords);

    std::thread melThread(audioToMels, ref(_settings), ref(state), ref(floatSamples),
                          ref(mels));
    std::thread featuresThread(melsToFeatures, ref(_settings), ref(state), ref(mels),
                               ref(features));

    std::vector<thread> wwThreads;
    for (size_t i = 0; i < numWakeWords; i++)
    {
        wwThreads.push_back(
            thread(featuresToOutput, std::ref(_settings), std::ref(state), i, std::ref(features), std::ref(wake_word_detected)));
    }

    // Block until ready
    const size_t numReadyExpected = 2 + numWakeWords;
    {
        std::unique_lock lockReady(state.mutReady);
        state.cvReady.wait(lockReady, [&state, numReadyExpected]
                           { return state.numReady == numReadyExpected; });
    }

    std::cerr << "[LOG] Ready" << '\n';
    if (SDL_Init(SDL_INIT_AUDIO) < 0)
    {
        std::cerr << "Failed to initialize SDL: " << SDL_GetError() << std::endl;
    } else {
        std::cout << "audio init success!" << '\n';
    }
    {
        int nDevices = SDL_GetNumAudioDevices(SDL_TRUE);
        fprintf(stderr, "%s: found %d capture devices:\n", __func__, nDevices);
        for (int i = 0; i < nDevices; i++) {
            fprintf(stderr, "%s:    - Capture device #%d: '%s'\n", __func__, i, SDL_GetAudioDeviceName(i, SDL_TRUE));
        }
    }
   
     // Set up audio specification

    SDL_AudioSpec capture_spec_requested;
    SDL_AudioSpec capture_spec_obtained;

    SDL_zero(capture_spec_requested);
    SDL_zero(capture_spec_obtained);
    capture_spec_requested.freq = SAMPLE_RATE;
    capture_spec_requested.format = AUDIO_S16SYS; // 16-bit signed PCM audio
    capture_spec_requested.channels = NUM_CHANNELS;
    capture_spec_requested.samples = SAMPLES_PER_CHUNK;
    capture_spec_requested.callback = audioCallback;

    // Audio buffer to store recorded audio
    std::vector<int16_t> audioBuffer;

    // Userdata to pass to the callback function
    capture_spec_requested.userdata = &audioBuffer;
    // Main loop
    SDL_AudioDeviceID m_dev_id_in = SDL_OpenAudioDevice(SDL_GetAudioDeviceName(1, SDL_TRUE), SDL_TRUE, &capture_spec_requested, &capture_spec_obtained, 0);
    if (!m_dev_id_in) {
        fprintf(stderr, "%s: couldn't open an audio device for capture: %s!\n", __func__, SDL_GetError());
        m_dev_id_in = 0;
    } else {
        fprintf(stderr, "%s: obtained spec for input device (SDL Id = %d):\n", __func__, m_dev_id_in);
        fprintf(stderr, "%s:     - sample rate:       %d\n",                   __func__, capture_spec_obtained.freq);
        fprintf(stderr, "%s:     - format:            %d (required: %d)\n",    __func__, capture_spec_obtained.format,
                capture_spec_requested.format);
        fprintf(stderr, "%s:     - channels:          %d (required: %d)\n",    __func__, capture_spec_obtained.channels,
                capture_spec_requested.channels);
        fprintf(stderr, "%s:     - samples per frame: %d\n",                   __func__, capture_spec_obtained.samples);
    }

    while (!do_exit)
    {
    // Start recording
    SDL_PauseAudioDevice(m_dev_id_in, 0);
    SDL_Delay(500);
    SDL_PauseAudioDevice(m_dev_id_in, 1);

        {
            std::unique_lock lockSamples{state.mutSamples};

            for (int16_t sample : audioBuffer)
            {
                // NOTE: we do NOT normalize here
                floatSamples.push_back((float)sample);
            }

            state.samplesReady = true;
            state.cvSamples.notify_one();
            audioBuffer.clear();
        }

    }

    // Signal mel thread that samples have been exhausted
    {
        std::unique_lock lockSamples{state.mutSamples};
        state.samplesExhausted = true;
        state.samplesReady = true;
        state.cvSamples.notify_one();
    }

    melThread.join();

    // Signal features thread that mels have been exhausted
    {
        std::unique_lock lockMels{state.mutMels};
        state.melsExhausted = true;
        state.melsReady = true;
        state.cvMels.notify_one();
    }
    featuresThread.join();

    // Signal wake word threads that features have been exhausted
    for (size_t i = 0; i < numWakeWords; i++)
    {
        std::unique_lock lockFeatures{state.mutFeatures[i]};
        state.featuresExhausted[i] = true;
        state.featuresReady[i] = true;
        state.cvFeatures[i].notify_one();
    }

    for (size_t i = 0; i < numWakeWords; i++)
    {
        wwThreads[i].join();
    }
}

void openwakeword_detector::init(std::string path_to_model)
{
    thr = std::thread(run_thread, path_to_model, std::ref(_wake_word_detected), std::ref(_do_exit));
}

uint8_t openwakeword_detector::detect_wakeword()
{
    if(_wake_word_detected) {
        _wake_word_detected = false;
        return true;
    } else {
        return false;
    }
}

openwakeword_detector::~openwakeword_detector() {
    _do_exit = true;
    thr.join();
}