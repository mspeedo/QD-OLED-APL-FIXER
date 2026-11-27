// Generic ReShade 6.x Shader Benchmark Add-on (ShaderBench.addon64)
// Works with ANY shader version (old, current, future) via external bench_config.txt.
// Measures exact hardware GPU time around ReShade effect execution using D3D11/D3D12 timestamp queries.

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <d3d12.h>
#include <wrl/client.h>

#include <reshade.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using Microsoft::WRL::ComPtr;

extern "C" __declspec(dllexport) const char *NAME = "QD-OLED Shader Benchmark";
extern "C" __declspec(dllexport) const char *DESCRIPTION =
    "Generic automated ReShade shader benchmark and compile-verification add-on driven by bench_config.txt.";

namespace
{
constexpr uint32_t kQueryRingSize = 6;

struct UniformSetting
{
    std::string kind; // "bool", "int", "uint", "float", or "auto"
    std::string name;
    std::string value;
};

struct BenchCase
{
    std::string name;
    std::map<std::string, std::string> defines;
    std::vector<UniformSetting> uniforms;
};

struct BenchConfig
{
    std::string effect = "EOTF_Boost_X36.fx";
    std::string technique = "EOTF_Boost_1D_APL_LUT";
    std::string outputCsv = "results.txt";
    std::string doneFile = "bench_done.flag";
    int warmupFrames = 60;
    int sampleFrames = 200;
    std::vector<BenchCase> cases;
};

struct D3D11QuerySlot
{
    ComPtr<ID3D11Query> disjoint;
    ComPtr<ID3D11Query> tsStart;
    ComPtr<ID3D11Query> tsEnd;
    bool open = false;
    bool submitted = false;
};

enum class BenchState
{
    Idle,
    WaitInitialLoad,
    SetupCase,
    WaitReload,
    ApplyUniforms,
    Warmup,
    Measure,
    Done
};

struct __declspec(uuid("7c4a9b12-8e34-4f1a-9d21-6c5e8a2f3b91")) BenchRuntimeData
{
    bool enabled = false;
    BenchConfig cfg;
    BenchState state = BenchState::Idle;

    size_t caseIdx = 0;
    std::map<std::string, std::string> appliedDefines;
    bool reloadedSinceRequest = false;
    bool initialReloadDone = false;
    uint32_t settleFrames = 0;
    ULONGLONG initStartTick = 0;
    ULONGLONG reloadStartTick = 0;
    uint32_t warmupLeft = 0;
    uint32_t measureFrames = 0;
    bool csvHeaderWritten = false;

    std::vector<double> samplesMs;

    // GPU timing backend
    reshade::api::device_api api = reshade::api::device_api::d3d11;
    bool gpuTimerReady = false;

    // D3D11 queries
    D3D11QuerySlot d11Slots[kQueryRingSize];
    uint64_t writeIdx = 0;
    uint64_t readIdx = 0;
    int activeSlot = -1;

    // D3D12 queries
    ComPtr<ID3D12QueryHeap> d12Heap;
    ComPtr<ID3D12Resource> d12Readback;
    uint64_t *d12Mapped = nullptr;
    uint64_t d12Freq = 0;

    // QPC fallback
    LARGE_INTEGER qpcFreq = {};
    LARGE_INTEGER qpcStart = {};
    bool qpcActive = false;
};

static std::string Trim(const std::string &s)
{
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

static bool LoadBenchConfig(const std::string &path, BenchConfig &out)
{
    std::ifstream f(path);
    if (!f.is_open())
        return false;

    BenchCase *curCase = nullptr;
    std::string line;
    while (std::getline(f, line))
    {
        std::string t = Trim(line);
        if (t.empty() || t[0] == '#' || t[0] == ';')
            continue;

        if (t.front() == '[' && t.back() == ']')
        {
            std::string sec = t.substr(1, t.size() - 2);
            if (sec.rfind("case:", 0) == 0)
            {
                out.cases.push_back(BenchCase{});
                curCase = &out.cases.back();
                curCase->name = Trim(sec.substr(5));
            }
            else
            {
                curCase = nullptr;
            }
            continue;
        }

        size_t eq = t.find('=');
        if (eq == std::string::npos)
            continue;

        std::string key = Trim(t.substr(0, eq));
        std::string val = Trim(t.substr(eq + 1));

        if (!curCase)
        {
            if (key == "effect") out.effect = val;
            else if (key == "technique") out.technique = val;
            else if (key == "output") out.outputCsv = val;
            else if (key == "done_file") out.doneFile = val;
            else if (key == "warmup") out.warmupFrames = std::max(1, std::atoi(val.c_str()));
            else if (key == "samples") out.sampleFrames = std::max(10, std::atoi(val.c_str()));
        }
        else
        {
            if (key.rfind("define:", 0) == 0)
            {
                curCase->defines[Trim(key.substr(7))] = val;
            }
            else if (key.rfind("bool:", 0) == 0)
            {
                curCase->uniforms.push_back({ "bool", Trim(key.substr(5)), val });
            }
            else if (key.rfind("int:", 0) == 0)
            {
                curCase->uniforms.push_back({ "int", Trim(key.substr(4)), val });
            }
            else if (key.rfind("uint:", 0) == 0)
            {
                curCase->uniforms.push_back({ "uint", Trim(key.substr(5)), val });
            }
            else if (key.rfind("float:", 0) == 0)
            {
                curCase->uniforms.push_back({ "float", Trim(key.substr(6)), val });
            }
            else if (key.rfind("uniform:", 0) == 0)
            {
                curCase->uniforms.push_back({ "auto", Trim(key.substr(8)), val });
            }
        }
    }

    return !out.cases.empty();
}

static void InitGpuTimers(reshade::api::effect_runtime *runtime, BenchRuntimeData &data)
{
    QueryPerformanceFrequency(&data.qpcFreq);
    data.api = runtime->get_device()->get_api();
    data.gpuTimerReady = false;

    if (data.api == reshade::api::device_api::d3d11)
    {
        ID3D11Device *dev = reinterpret_cast<ID3D11Device *>(runtime->get_device()->get_native());
        if (!dev) return;

        D3D11_QUERY_DESC dq = { D3D11_QUERY_TIMESTAMP_DISJOINT, 0 };
        D3D11_QUERY_DESC tq = { D3D11_QUERY_TIMESTAMP, 0 };
        bool ok = true;
        for (uint32_t i = 0; i < kQueryRingSize; ++i)
        {
            if (FAILED(dev->CreateQuery(&dq, &data.d11Slots[i].disjoint)) ||
                FAILED(dev->CreateQuery(&tq, &data.d11Slots[i].tsStart)) ||
                FAILED(dev->CreateQuery(&tq, &data.d11Slots[i].tsEnd)))
            {
                ok = false;
                break;
            }
        }
        data.gpuTimerReady = ok;
    }
    else if (data.api == reshade::api::device_api::d3d12)
    {
        ID3D12Device *dev = reinterpret_cast<ID3D12Device *>(runtime->get_device()->get_native());
        ID3D12CommandQueue *queue = reinterpret_cast<ID3D12CommandQueue *>(runtime->get_command_queue()->get_native());
        if (!dev || !queue) return;

        if (FAILED(queue->GetTimestampFrequency(&data.d12Freq)) || data.d12Freq == 0)
            return;

        D3D12_QUERY_HEAP_DESC qhDesc = {};
        qhDesc.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
        qhDesc.Count = kQueryRingSize * 2;
        if (FAILED(dev->CreateQueryHeap(&qhDesc, IID_PPV_ARGS(&data.d12Heap))))
            return;

        D3D12_HEAP_PROPERTIES hp = {};
        hp.Type = D3D12_HEAP_TYPE_READBACK;
        D3D12_RESOURCE_DESC rd = {};
        rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width = kQueryRingSize * 2 * sizeof(uint64_t);
        rd.Height = 1;
        rd.DepthOrArraySize = 1;
        rd.MipLevels = 1;
        rd.SampleDesc.Count = 1;
        rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        if (FAILED(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd,
                                                D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                IID_PPV_ARGS(&data.d12Readback))))
            return;

        void *mapped = nullptr;
        D3D12_RANGE readRange = { 0, static_cast<SIZE_T>(rd.Width) };
        if (FAILED(data.d12Readback->Map(0, &readRange, &mapped)))
            return;
        data.d12Mapped = static_cast<uint64_t *>(mapped);
        data.gpuTimerReady = true;
    }
}

static void ResetTimerRing(BenchRuntimeData &data)
{
    data.writeIdx = 0;
    data.readIdx = 0;
    data.activeSlot = -1;
    data.qpcActive = false;
    for (uint32_t i = 0; i < kQueryRingSize; ++i)
    {
        data.d11Slots[i].open = false;
        data.d11Slots[i].submitted = false;
    }
}

static std::string GetBufferColorSpaceString(reshade::api::effect_runtime *runtime)
{
    char buf[32] = {};
    if (runtime->get_preprocessor_definition("BUFFER_COLOR_SPACE", buf) && buf[0] != '\0')
        return std::string(buf);

    auto bb = runtime->get_current_back_buffer();
    if (bb.handle != 0)
    {
        auto desc = runtime->get_device()->get_resource_desc(bb);
        switch (desc.texture.format)
        {
        case reshade::api::format::r10g10b10a2_unorm:    return "10bit_PQ";
        case reshade::api::format::r16g16b16a16_float:  return "16f_scRGB";
        case reshade::api::format::r8g8b8a8_unorm:
        case reshade::api::format::r8g8b8a8_unorm_srgb:
        case reshade::api::format::b8g8r8a8_unorm:
        case reshade::api::format::b8g8r8a8_unorm_srgb: return "8bit_SDR";
        default: break;
        }
    }
    return "unknown";
}

static std::string GetSystemDecimalSeparator()
{
    char buf[16] = {};
    if (GetLocaleInfoA(LOCALE_USER_DEFAULT, LOCALE_SDECIMAL, buf, static_cast<int>(sizeof(buf))) > 1 && buf[0] != '\0')
        return std::string(buf);
    return ".";
}

static std::string FormatLocaleDouble(double v, const std::string &decSep)
{
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.4f", v);
    std::string s(buf);
    if (decSep != ".")
    {
        size_t pos = s.find('.');
        if (pos != std::string::npos)
            s.replace(pos, 1, decSep);
    }
    return s;
}

static void WriteCsvRow(
    reshade::api::effect_runtime *runtime,
    BenchRuntimeData &data,
    const std::string &caseName,
    const std::string &status,
    double medianMs,
    double meanMs,
    double p95Ms,
    double minMs,
    double maxMs,
    size_t validSamples)
{
    const std::string sep = "\t";
    const std::string dec = GetSystemDecimalSeparator();

    std::ios::openmode mode = data.csvHeaderWritten ? (std::ios::out | std::ios::app) : (std::ios::out | std::ios::trunc);
    std::ofstream out(data.cfg.outputCsv, mode);
    if (!out.is_open())
        return;

    if (!data.csvHeaderWritten)
    {
        out << "case" << sep << "status" << sep << "color_space" << sep << "width" << sep << "height" << sep
            << "timer" << sep << "samples" << sep << "median_ms" << sep << "mean_ms" << sep
            << "p95_ms" << sep << "min_ms" << sep << "max_ms\n";
        data.csvHeaderWritten = true;
    }

    uint32_t w = 0, h = 0;
    runtime->get_screenshot_width_and_height(&w, &h);
    const char *timerName = data.gpuTimerReady
        ? (data.api == reshade::api::device_api::d3d12 ? "d3d12_gpu" : "d3d11_gpu")
        : "cpu_qpc";

    std::ostringstream oss;
    oss << caseName << sep
        << status << sep
        << GetBufferColorSpaceString(runtime) << sep
        << w << sep << h << sep
        << timerName << sep
        << validSamples << sep
        << FormatLocaleDouble(medianMs, dec) << sep
        << FormatLocaleDouble(meanMs, dec) << sep
        << FormatLocaleDouble(p95Ms, dec) << sep
        << FormatLocaleDouble(minMs, dec) << sep
        << FormatLocaleDouble(maxMs, dec) << "\n";

    std::string line = oss.str();
    out << line;
    out.flush();

    reshade::log::message(reshade::log::level::info, line.c_str());
}

static void WriteDoneFlag(const std::string &path)
{
    std::ofstream f(path, std::ios::out | std::ios::trunc);
    f << "done\n";
    f.flush();
}

static void ApplyUniformSetting(
    reshade::api::effect_runtime *runtime,
    const std::string &effectName,
    const UniformSetting &u)
{
    auto var = runtime->find_uniform_variable(effectName.c_str(), u.name.c_str());
    if (var.handle == 0)
        return; // Gracefully skip controls that do not exist in older shader versions

    if (u.kind == "bool")
    {
        bool b = (u.value == "1" || u.value == "true" || u.value == "TRUE");
        runtime->set_uniform_value_bool(var, b);
    }
    else if (u.kind == "int")
    {
        int32_t v = static_cast<int32_t>(std::atoi(u.value.c_str()));
        runtime->set_uniform_value_int(var, v);
    }
    else if (u.kind == "uint")
    {
        uint32_t v = static_cast<uint32_t>(std::strtoul(u.value.c_str(), nullptr, 10));
        runtime->set_uniform_value_uint(var, v);
    }
    else if (u.kind == "float")
    {
        float v = static_cast<float>(std::atof(u.value.c_str()));
        runtime->set_uniform_value_float(var, v);
    }
    else
    {
        reshade::api::format baseType = reshade::api::format::unknown;
        runtime->get_uniform_variable_type(var, &baseType);
        if (baseType == reshade::api::format::r32_float)
            runtime->set_uniform_value_float(var, static_cast<float>(std::atof(u.value.c_str())));
        else if (baseType == reshade::api::format::r32_sint)
            runtime->set_uniform_value_int(var, static_cast<int32_t>(std::atoi(u.value.c_str())));
        else if (baseType == reshade::api::format::r32_uint)
            runtime->set_uniform_value_uint(var, static_cast<uint32_t>(std::strtoul(u.value.c_str(), nullptr, 10)));
        else
            runtime->set_uniform_value_bool(var, u.value == "1" || u.value == "true");
    }
}

static void HarvestTimers(reshade::api::effect_runtime *runtime, BenchRuntimeData &data)
{
    if (!data.gpuTimerReady)
        return;

    if (data.api == reshade::api::device_api::d3d11)
    {
        ID3D11Device *dev = reinterpret_cast<ID3D11Device *>(runtime->get_device()->get_native());
        if (!dev) return;
        ComPtr<ID3D11DeviceContext> ctx;
        dev->GetImmediateContext(&ctx);
        if (!ctx) return;

        while (data.writeIdx - data.readIdx >= 3)
        {
            uint32_t slotIdx = static_cast<uint32_t>(data.readIdx % kQueryRingSize);
            D3D11QuerySlot &slot = data.d11Slots[slotIdx];
            if (!slot.submitted)
            {
                ++data.readIdx;
                continue;
            }

            D3D11_QUERY_DATA_TIMESTAMP_DISJOINT dj = {};
            uint64_t t0 = 0, t1 = 0;
            HRESULT hr0 = ctx->GetData(slot.disjoint.Get(), &dj, sizeof(dj), D3D11_ASYNC_GETDATA_DONOTFLUSH);
            HRESULT hr1 = ctx->GetData(slot.tsStart.Get(), &t0, sizeof(t0), D3D11_ASYNC_GETDATA_DONOTFLUSH);
            HRESULT hr2 = ctx->GetData(slot.tsEnd.Get(), &t1, sizeof(t1), D3D11_ASYNC_GETDATA_DONOTFLUSH);
            if (hr0 != S_OK || hr1 != S_OK || hr2 != S_OK)
                break;

            slot.submitted = false;
            ++data.readIdx;

            if (!dj.Disjoint && dj.Frequency > 0 && t1 >= t0)
            {
                double ms = static_cast<double>(t1 - t0) * 1000.0 / static_cast<double>(dj.Frequency);
                data.samplesMs.push_back(ms);
            }
        }
    }
    else if (data.api == reshade::api::device_api::d3d12 && data.d12Mapped && data.d12Freq > 0)
    {
        while (data.writeIdx - data.readIdx >= 4)
        {
            uint32_t slotIdx = static_cast<uint32_t>(data.readIdx % kQueryRingSize);
            uint64_t t0 = data.d12Mapped[slotIdx * 2 + 0];
            uint64_t t1 = data.d12Mapped[slotIdx * 2 + 1];
            ++data.readIdx;
            if (t1 >= t0)
            {
                double ms = static_cast<double>(t1 - t0) * 1000.0 / static_cast<double>(data.d12Freq);
                data.samplesMs.push_back(ms);
            }
        }
    }
}

static void FinishCurrentCase(reshade::api::effect_runtime *runtime, BenchRuntimeData &data)
{
    const BenchCase &bc = data.cfg.cases[data.caseIdx];
    if (data.samplesMs.empty())
    {
        std::printf("[ShaderBench] [%zu/%zu] %-30s -> NO_SAMPLES\n",
                    data.caseIdx + 1, data.cfg.cases.size(), bc.name.c_str());
        std::fflush(stdout);
        WriteCsvRow(runtime, data, bc.name, "NO_SAMPLES", 0, 0, 0, 0, 0, 0);
    }
    else
    {
        std::sort(data.samplesMs.begin(), data.samplesMs.end());
        size_t n = data.samplesMs.size();
        double median = (n % 2 == 1)
            ? data.samplesMs[n / 2]
            : 0.5 * (data.samplesMs[n / 2 - 1] + data.samplesMs[n / 2]);
        double mean = std::accumulate(data.samplesMs.begin(), data.samplesMs.end(), 0.0) / static_cast<double>(n);
        size_t p95Idx = std::min(n - 1, static_cast<size_t>(std::floor(0.95 * static_cast<double>(n))));
        double p95 = data.samplesMs[p95Idx];
        double minV = data.samplesMs.front();
        double maxV = data.samplesMs.back();
        std::printf("[ShaderBench] [%zu/%zu] %-30s -> median=%.4f ms  mean=%.4f ms  p95=%.4f ms  (%zu samples)\n",
                    data.caseIdx + 1, data.cfg.cases.size(), bc.name.c_str(), median, mean, p95, n);
        std::fflush(stdout);
        WriteCsvRow(runtime, data, bc.name, "OK", median, mean, p95, minV, maxV, n);
    }

    ++data.caseIdx;
    if (data.caseIdx >= data.cfg.cases.size())
    {
        std::printf("[ShaderBench] All %zu cases complete! Saved %s\n",
                    data.cfg.cases.size(), data.cfg.outputCsv.c_str());
        std::fflush(stdout);
        data.state = BenchState::Done;
        WriteDoneFlag(data.cfg.doneFile);
    }
    else
    {
        data.state = BenchState::SetupCase;
    }
}

static void OnInitEffectRuntime(reshade::api::effect_runtime *runtime)
{
    auto &data = *runtime->create_private_data<BenchRuntimeData>();

    char autoEnv[16] = {};
    DWORD envLen = GetEnvironmentVariableA("SHADER_BENCH_AUTO", autoEnv, static_cast<DWORD>(sizeof(autoEnv)));
    if (envLen == 0 || std::strcmp(autoEnv, "1") != 0)
    {
        data.enabled = false;
        reshade::log::message(reshade::log::level::info, "[ShaderBench] Interactive mode (SHADER_BENCH_AUTO != 1); add-on idle.");
        return;
    }

    if (!LoadBenchConfig("bench_config.txt", data.cfg))
    {
        data.enabled = false;
        return;
    }

    data.enabled = true;
    InitGpuTimers(runtime, data);
    data.initStartTick = GetTickCount64();
    data.initialReloadDone = false;
    data.settleFrames = 0;
    data.state = BenchState::WaitInitialLoad;
    std::printf("[ShaderBench] Loaded bench_config.txt (%zu cases). Waiting for ReShade to finish compiling shaders...\n",
                data.cfg.cases.size());
    std::fflush(stdout);
    reshade::log::message(reshade::log::level::info, "[ShaderBench] Loaded bench_config.txt; waiting for ReShade shader compilation.");
}

static void OnDestroyEffectRuntime(reshade::api::effect_runtime *runtime)
{
    auto *data = runtime->get_private_data<BenchRuntimeData>();
    if (data && data->d12Readback && data->d12Mapped)
    {
        data->d12Readback->Unmap(0, nullptr);
        data->d12Mapped = nullptr;
    }
    runtime->destroy_private_data<BenchRuntimeData>();
}

static void OnReloadedEffects(reshade::api::effect_runtime *runtime)
{
    auto *data = runtime->get_private_data<BenchRuntimeData>();
    if (!data || !data->enabled)
        return;

    data->reloadedSinceRequest = true;
    data->initialReloadDone = true;
    data->settleFrames = 0;
}

static void OnBeginEffects(
    reshade::api::effect_runtime *runtime,
    reshade::api::command_list *cmd_list,
    reshade::api::resource_view,
    reshade::api::resource_view)
{
    auto *data = runtime->get_private_data<BenchRuntimeData>();
    if (!data || !data->enabled || data->state != BenchState::Measure)
        return;

    if (data->gpuTimerReady && (data->writeIdx - data->readIdx) < (kQueryRingSize - 1))
    {
        int slotIdx = static_cast<int>(data->writeIdx % kQueryRingSize);
        if (data->api == reshade::api::device_api::d3d11)
        {
            ID3D11DeviceContext *ctx = reinterpret_cast<ID3D11DeviceContext *>(cmd_list->get_native());
            if (ctx)
            {
                D3D11QuerySlot &slot = data->d11Slots[slotIdx];
                ctx->Begin(slot.disjoint.Get());
                ctx->End(slot.tsStart.Get());
                slot.open = true;
                data->activeSlot = slotIdx;
            }
        }
        else if (data->api == reshade::api::device_api::d3d12 && data->d12Heap)
        {
            ID3D12GraphicsCommandList *cl = reinterpret_cast<ID3D12GraphicsCommandList *>(cmd_list->get_native());
            if (cl)
            {
                cl->EndQuery(data->d12Heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, static_cast<UINT>(slotIdx * 2 + 0));
                data->activeSlot = slotIdx;
            }
        }
    }
    else if (!data->gpuTimerReady)
    {
        QueryPerformanceCounter(&data->qpcStart);
        data->qpcActive = true;
    }
}

static void OnFinishEffects(
    reshade::api::effect_runtime *runtime,
    reshade::api::command_list *cmd_list,
    reshade::api::resource_view,
    reshade::api::resource_view)
{
    auto *data = runtime->get_private_data<BenchRuntimeData>();
    if (!data || !data->enabled || data->state != BenchState::Measure)
        return;

    if (data->gpuTimerReady && data->activeSlot >= 0)
    {
        int slotIdx = data->activeSlot;
        data->activeSlot = -1;

        if (data->api == reshade::api::device_api::d3d11)
        {
            ID3D11DeviceContext *ctx = reinterpret_cast<ID3D11DeviceContext *>(cmd_list->get_native());
            if (ctx && data->d11Slots[slotIdx].open)
            {
                D3D11QuerySlot &slot = data->d11Slots[slotIdx];
                ctx->End(slot.tsEnd.Get());
                ctx->End(slot.disjoint.Get());
                slot.open = false;
                slot.submitted = true;
                ++data->writeIdx;
            }
        }
        else if (data->api == reshade::api::device_api::d3d12 && data->d12Heap && data->d12Readback)
        {
            ID3D12GraphicsCommandList *cl = reinterpret_cast<ID3D12GraphicsCommandList *>(cmd_list->get_native());
            if (cl)
            {
                cl->EndQuery(data->d12Heap.Get(), D3D12_QUERY_TYPE_TIMESTAMP, static_cast<UINT>(slotIdx * 2 + 1));
                cl->ResolveQueryData(
                    data->d12Heap.Get(),
                    D3D12_QUERY_TYPE_TIMESTAMP,
                    static_cast<UINT>(slotIdx * 2),
                    2,
                    data->d12Readback.Get(),
                    static_cast<UINT64>(slotIdx) * 2 * sizeof(uint64_t));
                ++data->writeIdx;
            }
        }
    }
    else if (!data->gpuTimerReady && data->qpcActive && data->qpcFreq.QuadPart > 0)
    {
        LARGE_INTEGER now = {};
        QueryPerformanceCounter(&now);
        double ms = static_cast<double>(now.QuadPart - data->qpcStart.QuadPart) * 1000.0 /
                    static_cast<double>(data->qpcFreq.QuadPart);
        data->samplesMs.push_back(ms);
        data->qpcActive = false;
    }
}

static void OnPresent(reshade::api::effect_runtime *runtime)
{
    auto *data = runtime->get_private_data<BenchRuntimeData>();
    if (!data || !data->enabled || data->state == BenchState::Done || data->state == BenchState::Idle)
        return;

    if (data->state == BenchState::WaitInitialLoad)
    {
        auto tech = runtime->find_technique(data->cfg.effect.c_str(), data->cfg.technique.c_str());
        bool techLoaded = (tech.handle != 0);
        ULONGLONG elapsedMs = GetTickCount64() - data->initStartTick;

        // Wait until ReShade finishes compiling shaders (reshade_reloaded_effects fired AND technique exists),
        // or if effects were already loaded before init, wait at least 2 seconds with technique present.
        if ((data->initialReloadDone && techLoaded) || (techLoaded && elapsedMs >= 2000ULL))
        {
            if (++data->settleFrames >= 60)
            {
                std::printf("[ShaderBench] Shader compiled and ready! Starting benchmark (%zu cases)...\n",
                            data->cfg.cases.size());
                std::fflush(stdout);
                data->state = BenchState::SetupCase;
            }
        }
        else if (elapsedMs > 60000ULL)
        {
            std::printf("[ShaderBench] ERROR: Timed out waiting 60s for %s (%s) to compile.\n",
                        data->cfg.effect.c_str(), data->cfg.technique.c_str());
            std::fflush(stdout);
            WriteCsvRow(runtime, *data, "INITIAL_COMPILE", "COMPILE_FAIL", 0, 0, 0, 0, 0, 0);
            data->state = BenchState::Done;
            WriteDoneFlag(data->cfg.doneFile);
        }
        return;
    }

    if (data->state == BenchState::SetupCase)
    {
        if (data->caseIdx >= data->cfg.cases.size())
        {
            data->state = BenchState::Done;
            WriteDoneFlag(data->cfg.doneFile);
            return;
        }

        const BenchCase &bc = data->cfg.cases[data->caseIdx];
        bool needReload = (bc.defines != data->appliedDefines);
        if (needReload)
        {
            for (const auto &kv : data->appliedDefines)
            {
                if (bc.defines.find(kv.first) == bc.defines.end())
                    runtime->set_preprocessor_definition(kv.first.c_str(), nullptr);
            }
            for (const auto &kv : bc.defines)
            {
                runtime->set_preprocessor_definition(kv.first.c_str(), kv.second.c_str());
            }
            data->appliedDefines = bc.defines;
            data->reloadedSinceRequest = false;
            data->settleFrames = 0;
            data->reloadStartTick = GetTickCount64();
            runtime->reload_effect_next_frame(data->cfg.effect.c_str());
            data->state = BenchState::WaitReload;
            return;
        }

        data->state = BenchState::ApplyUniforms;
    }

    if (data->state == BenchState::WaitReload)
    {
        bool timedOut = (GetTickCount64() - data->reloadStartTick) > 60000ULL;
        if (data->reloadedSinceRequest)
        {
            if (++data->settleFrames >= 30)
                data->state = BenchState::ApplyUniforms;
        }
        else if (timedOut)
        {
            data->state = BenchState::ApplyUniforms;
        }
        return;
    }

    if (data->state == BenchState::ApplyUniforms)
    {
        const BenchCase &bc = data->cfg.cases[data->caseIdx];
        runtime->set_effects_state(true);

        // Disable all loaded techniques, then enable only our target technique
        runtime->enumerate_techniques(nullptr, [](reshade::api::effect_runtime *rt, reshade::api::effect_technique t) {
            rt->set_technique_state(t, false);
        });

        auto tech = runtime->find_technique(data->cfg.effect.c_str(), data->cfg.technique.c_str());
        if (tech.handle == 0)
        {
            WriteCsvRow(runtime, *data, bc.name, "COMPILE_FAIL", 0, 0, 0, 0, 0, 0);
            ++data->caseIdx;
            data->state = (data->caseIdx >= data->cfg.cases.size()) ? BenchState::Done : BenchState::SetupCase;
            if (data->state == BenchState::Done)
                WriteDoneFlag(data->cfg.doneFile);
            return;
        }

        runtime->set_technique_state(tech, true);

        // Reset every uniform in this effect to its default before applying case overrides
        runtime->enumerate_uniform_variables(data->cfg.effect.c_str(),
            [](reshade::api::effect_runtime *rt, reshade::api::effect_uniform_variable v) {
                rt->reset_uniform_value(v);
            });

        for (const auto &u : bc.uniforms)
            ApplyUniformSetting(runtime, data->cfg.effect, u);

        data->samplesMs.clear();
        data->samplesMs.reserve(static_cast<size_t>(data->cfg.sampleFrames) + 16);
        ResetTimerRing(*data);
        data->warmupLeft = static_cast<uint32_t>(data->cfg.warmupFrames);
        data->measureFrames = 0;
        data->state = BenchState::Warmup;
        return;
    }

    if (data->state == BenchState::Warmup)
    {
        if (data->warmupLeft > 0)
            --data->warmupLeft;
        if (data->warmupLeft == 0)
            data->state = BenchState::Measure;
        return;
    }

    if (data->state == BenchState::Measure)
    {
        HarvestTimers(runtime, *data);
        ++data->measureFrames;

        uint32_t maxFrames = static_cast<uint32_t>(data->cfg.sampleFrames) * 3 + 60;
        if (static_cast<int>(data->samplesMs.size()) >= data->cfg.sampleFrames || data->measureFrames >= maxFrames)
        {
            FinishCurrentCase(runtime, *data);
        }
    }
}
} // namespace

BOOL APIENTRY DllMain(HMODULE hModule, DWORD fdwReason, LPVOID)
{
    switch (fdwReason)
    {
    case DLL_PROCESS_ATTACH:
        if (!reshade::register_addon(hModule))
            return FALSE;
        reshade::register_event<reshade::addon_event::init_effect_runtime>(OnInitEffectRuntime);
        reshade::register_event<reshade::addon_event::destroy_effect_runtime>(OnDestroyEffectRuntime);
        reshade::register_event<reshade::addon_event::reshade_reloaded_effects>(OnReloadedEffects);
        reshade::register_event<reshade::addon_event::reshade_begin_effects>(OnBeginEffects);
        reshade::register_event<reshade::addon_event::reshade_finish_effects>(OnFinishEffects);
        reshade::register_event<reshade::addon_event::reshade_present>(OnPresent);
        break;
    case DLL_PROCESS_DETACH:
        reshade::unregister_addon(hModule);
        break;
    }
    return TRUE;
}
