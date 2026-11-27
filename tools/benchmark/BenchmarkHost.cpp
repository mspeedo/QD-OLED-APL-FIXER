// Lightweight deterministic D3D11 HDR/SDR swapchain host for ReShade shader benchmarking.
// Builds with: cl /O2 /EHsc /W3 BenchmarkHost.cpp d3d11.lib dxgi.lib d3dcompiler.lib user32.lib

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d11.h>
#include <dxgi1_5.h>
#include <d3dcompiler.h>
#include <wrl/client.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

using Microsoft::WRL::ComPtr;

struct HostConfig
{
    std::string mode = "pq";      // pq | scrgb | sdr
    int width = 3440;
    int height = 1440;
    std::string doneFile = "bench_done.flag";
    int timeoutSec = 300;
};

struct SceneCB
{
    int modeId; // 0 = sdr, 1 = scrgb, 2 = pq
    float framePhase;
    float pad0;
    float pad1;
};

static const char *g_SceneShaderHLSL = R"HLSL(
cbuffer SceneCB : register(b0)
{
    int g_ModeId;
    float g_FramePhase;
    float g_Pad0;
    float g_Pad1;
};

struct VSOut
{
    float4 pos : SV_Position;
    float2 uv  : TEXCOORD0;
};

VSOut VSMain(uint id : SV_VertexID)
{
    VSOut o;
    o.uv = float2((id == 2) ? 2.0 : 0.0, (id == 1) ? 2.0 : 0.0);
    o.pos = float4(o.uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
    return o;
}

float3 LinearToPQ(float3 lin)
{
    const float m1 = 0.1593017578125;
    const float m2 = 78.84375;
    const float c1 = 0.8359375;
    const float c2 = 18.8515625;
    const float c3 = 18.6875;
    float3 ym = pow(saturate(lin), m1);
    return pow((c1 + c2 * ym) / (1.0 + c3 * ym), m2);
}

float3 LinearToSRGB(float3 c)
{
    c = saturate(c);
    return lerp(c * 12.92, 1.055 * pow(max(c, 1e-6), 1.0 / 2.4) - 0.055, step(0.0031308, c));
}

// Deterministic mid-APL (~15% APL) scene exercising:
// - dark shadows (2-20 nits)
// - midtones (50-250 nits)
// - saturated color bands (color-preserving boost mode)
// - high-contrast 600-1400 nit highlight details (BT.2390 shoulder + bilateral highlight detail)
float4 PSMain(VSOut i) : SV_Target
{
    float2 uv = i.uv;

    // Sky/ground luminance gradient (~15% average APL relative to 1000 nits)
    float skyMask = smoothstep(0.60, 0.35, uv.y);
    float3 nits = lerp(float3(18.0, 22.0, 26.0), float3(185.0, 210.0, 255.0), skyMask);

    // Subtle spatial detail so bilateral edges are non-trivial everywhere
    float fine = 0.5 + 0.5 * sin(uv.x * 160.0) * cos(uv.y * 90.0);
    nits *= lerp(0.88, 1.12, fine);

    // Saturated colored patches (red / green / blue / amber) to exercise color-preserving boost
    if (uv.y > 0.65 && uv.y < 0.90)
    {
        if (uv.x > 0.08 && uv.x < 0.28)
            nits = float3(260.0, 28.0, 22.0);
        else if (uv.x > 0.32 && uv.x < 0.52)
            nits = float3(30.0, 240.0, 35.0);
        else if (uv.x > 0.56 && uv.x < 0.76)
            nits = float3(25.0, 65.0, 270.0);
        else if (uv.x > 0.80 && uv.x < 0.94)
            nits = float3(320.0, 175.0, 25.0);
    }

    // High-contrast specular highlight cluster (600-1400 nits with crisp detail edges)
    float2 dSun = (uv - float2(0.30, 0.24)) * float2(1.8, 1.0);
    float rSun = length(dSun);
    if (rSun < 0.22)
    {
        float falloff = smoothstep(0.22, 0.02, rSun);
        float detail = 0.72 + 0.28 * sin(uv.x * 380.0 + g_FramePhase * 0.001) * cos(uv.y * 240.0);
        nits += float3(1150.0, 1020.0, 820.0) * falloff * detail;
    }

    // Secondary bright cloud rim highlights (~500-950 nits)
    float2 dCloud = (uv - float2(0.72, 0.28)) * float2(1.4, 2.2);
    float rCloud = length(dCloud);
    if (rCloud < 0.28)
    {
        float rim = smoothstep(0.28, 0.05, rCloud);
        float tex = 0.65 + 0.35 * sin(uv.x * 260.0) * sin(uv.y * 260.0);
        nits += float3(680.0, 710.0, 760.0) * rim * tex;
    }

    if (g_ModeId == 2) // HDR10 PQ (Rec.2020 approx)
    {
        return float4(LinearToPQ(nits / 10000.0), 1.0);
    }
    else if (g_ModeId == 1) // scRGB linear (1.0 = 80 nits)
    {
        return float4(nits / 80.0, 1.0);
    }
    else // SDR sRGB (1.0 = 200 nits display-referred)
    {
        return float4(LinearToSRGB(nits / 250.0), 1.0);
    }
}
)HLSL";

static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    if (msg == WM_DESTROY)
    {
        PostQuitMessage(0);
        return 0;
    }
    if (msg == WM_KEYDOWN && wParam == VK_ESCAPE)
    {
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcW(hwnd, msg, wParam, lParam);
}

static HostConfig ParseArgs(int argc, char **argv)
{
    HostConfig cfg;
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc)
            cfg.mode = argv[++i];
        else if (std::strcmp(argv[i], "--width") == 0 && i + 1 < argc)
            cfg.width = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--height") == 0 && i + 1 < argc)
            cfg.height = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--done-file") == 0 && i + 1 < argc)
            cfg.doneFile = argv[++i];
        else if (std::strcmp(argv[i], "--timeout") == 0 && i + 1 < argc)
            cfg.timeoutSec = std::atoi(argv[++i]);
    }
    if (cfg.width < 320) cfg.width = 320;
    if (cfg.height < 240) cfg.height = 240;
    return cfg;
}

static bool FileExists(const std::string &path)
{
    DWORD attr = GetFileAttributesA(path.c_str());
    return (attr != INVALID_FILE_ATTRIBUTES) && ((attr & FILE_ATTRIBUTE_DIRECTORY) == 0);
}

int main(int argc, char **argv)
{
    HostConfig cfg = ParseArgs(argc, argv);

    // Clear stale completion flag before starting
    DeleteFileA(cfg.doneFile.c_str());

    std::printf("[BenchmarkHost] Mode=%s  Resolution=%dx%d  Timeout=%ds\n",
                cfg.mode.c_str(), cfg.width, cfg.height, cfg.timeoutSec);
    std::fflush(stdout);

    HINSTANCE hInst = GetModuleHandleW(nullptr);
    WNDCLASSEXW wc = {};
    wc.cbSize = sizeof(wc);
    wc.style = CS_OWNDC;
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInst;
    wc.lpszClassName = L"QDOLEDShaderBenchHost";
    RegisterClassExW(&wc);

    // Use a borderless popup window sized to the requested backbuffer resolution
    // so ReShade sees the exact target BUFFER_WIDTH x BUFFER_HEIGHT (e.g. 3440x1440).
    HWND hwnd = CreateWindowExW(
        0,
        wc.lpszClassName,
        L"QD-OLED Shader Benchmark Host (ESC to abort)",
        WS_POPUP | WS_VISIBLE,
        0, 0, cfg.width, cfg.height,
        nullptr, nullptr, hInst, nullptr);

    if (!hwnd)
    {
        std::fprintf(stderr, "[BenchmarkHost] ERROR: CreateWindowExW failed.\n");
        return 1;
    }

    D3D_FEATURE_LEVEL featureLevel = D3D_FEATURE_LEVEL_11_0;
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    HRESULT hr = D3D11CreateDevice(
        nullptr,
        D3D_DRIVER_TYPE_HARDWARE,
        nullptr,
        0,
        &featureLevel,
        1,
        D3D11_SDK_VERSION,
        &device,
        nullptr,
        &context);
    if (FAILED(hr))
    {
        std::fprintf(stderr, "[BenchmarkHost] ERROR: D3D11CreateDevice failed (0x%08lX).\n", hr);
        return 2;
    }

    ComPtr<IDXGIFactory2> factory2;
    hr = CreateDXGIFactory1(IID_PPV_ARGS(&factory2));
    if (FAILED(hr))
    {
        std::fprintf(stderr, "[BenchmarkHost] ERROR: CreateDXGIFactory1 failed (0x%08lX).\n", hr);
        return 3;
    }

    BOOL allowTearing = FALSE;
    ComPtr<IDXGIFactory5> factory5;
    if (SUCCEEDED(factory2.As(&factory5)))
    {
        factory5->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING, &allowTearing, sizeof(allowTearing));
    }

    DXGI_FORMAT bbFormat = DXGI_FORMAT_R10G10B10A2_UNORM;
    DXGI_COLOR_SPACE_TYPE colorSpace = DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020;
    int modeId = 2;

    if (cfg.mode == "scrgb")
    {
        bbFormat = DXGI_FORMAT_R16G16B16A16_FLOAT;
        colorSpace = DXGI_COLOR_SPACE_RGB_FULL_G10_NONE_P709;
        modeId = 1;
    }
    else if (cfg.mode == "sdr")
    {
        bbFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
        colorSpace = DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709;
        modeId = 0;
    }

    DXGI_SWAP_CHAIN_DESC1 scDesc = {};
    scDesc.Width = static_cast<UINT>(cfg.width);
    scDesc.Height = static_cast<UINT>(cfg.height);
    scDesc.Format = bbFormat;
    scDesc.SampleDesc.Count = 1;
    scDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    scDesc.BufferCount = 2;
    scDesc.Scaling = DXGI_SCALING_STRETCH;
    scDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
    scDesc.AlphaMode = DXGI_ALPHA_MODE_IGNORE;
    scDesc.Flags = allowTearing ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;

    ComPtr<IDXGISwapChain1> swapChain1;
    hr = factory2->CreateSwapChainForHwnd(device.Get(), hwnd, &scDesc, nullptr, nullptr, &swapChain1);
    if (FAILED(hr))
    {
        std::fprintf(stderr, "[BenchmarkHost] ERROR: CreateSwapChainForHwnd failed (0x%08lX).\n", hr);
        return 4;
    }

    ComPtr<IDXGISwapChain3> swapChain3;
    if (SUCCEEDED(swapChain1.As(&swapChain3)))
    {
        UINT support = 0;
        hr = swapChain3->CheckColorSpaceSupport(colorSpace, &support);
        if (SUCCEEDED(hr) && (support & DXGI_SWAP_CHAIN_COLOR_SPACE_SUPPORT_FLAG_PRESENT))
        {
            HRESULT csHr = swapChain3->SetColorSpace1(colorSpace);
            if (FAILED(csHr))
                std::printf("[BenchmarkHost] WARN: SetColorSpace1 returned 0x%08lX (is Windows HDR enabled?).\n", csHr);
            else
                std::printf("[BenchmarkHost] Swapchain color space configured cleanly.\n");
        }
        else
        {
            // Still try setting it so ReShade's DXGI hook sees the requested swapchain color space
            HRESULT csHr = swapChain3->SetColorSpace1(colorSpace);
            std::printf("[BenchmarkHost] WARN: Display did not advertise support flag for mode=%s (SetColorSpace1=0x%08lX).\n",
                        cfg.mode.c_str(), csHr);
        }
    }

    ComPtr<ID3D11Texture2D> backBuffer;
    hr = swapChain1->GetBuffer(0, IID_PPV_ARGS(&backBuffer));
    if (FAILED(hr)) return 5;

    ComPtr<ID3D11RenderTargetView> rtv;
    hr = device->CreateRenderTargetView(backBuffer.Get(), nullptr, &rtv);
    if (FAILED(hr)) return 6;

    // Compile built-in synthetic test scene shader
    ComPtr<ID3DBlob> vsBlob, psBlob, errBlob;
    hr = D3DCompile(g_SceneShaderHLSL, std::strlen(g_SceneShaderHLSL), "Scene.hlsl", nullptr, nullptr,
                    "VSMain", "vs_5_0", 0, 0, &vsBlob, &errBlob);
    if (FAILED(hr))
    {
        if (errBlob) std::fprintf(stderr, "VS compile error: %s\n", (const char *)errBlob->GetBufferPointer());
        return 7;
    }
    hr = D3DCompile(g_SceneShaderHLSL, std::strlen(g_SceneShaderHLSL), "Scene.hlsl", nullptr, nullptr,
                    "PSMain", "ps_5_0", 0, 0, &psBlob, &errBlob);
    if (FAILED(hr))
    {
        if (errBlob) std::fprintf(stderr, "PS compile error: %s\n", (const char *)errBlob->GetBufferPointer());
        return 8;
    }

    ComPtr<ID3D11VertexShader> vs;
    ComPtr<ID3D11PixelShader> ps;
    device->CreateVertexShader(vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), nullptr, &vs);
    device->CreatePixelShader(psBlob->GetBufferPointer(), psBlob->GetBufferSize(), nullptr, &ps);

    D3D11_BUFFER_DESC cbDesc = {};
    cbDesc.ByteWidth = sizeof(SceneCB);
    cbDesc.Usage = D3D11_USAGE_DYNAMIC;
    cbDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    cbDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    ComPtr<ID3D11Buffer> cb;
    device->CreateBuffer(&cbDesc, nullptr, &cb);

    D3D11_VIEWPORT vp = {};
    vp.Width = static_cast<float>(cfg.width);
    vp.Height = static_cast<float>(cfg.height);
    vp.MaxDepth = 1.0f;

    ULONGLONG startTick = GetTickCount64();
    uint64_t frameCount = 0;
    MSG msg = {};
    bool running = true;
    int exitCode = 0;

    while (running)
    {
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE))
        {
            if (msg.message == WM_QUIT)
            {
                running = false;
                exitCode = 10; // Aborted by user
                break;
            }
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
        if (!running) break;

        if (FileExists(cfg.doneFile))
        {
            std::printf("[BenchmarkHost] Completion flag detected after %llu frames. Exiting cleanly.\n",
                        (unsigned long long)frameCount);
            exitCode = 0;
            break;
        }

        if (cfg.timeoutSec > 0 && (GetTickCount64() - startTick) > static_cast<ULONGLONG>(cfg.timeoutSec) * 1000ULL)
        {
            std::fprintf(stderr, "[BenchmarkHost] ERROR: Timed out after %d seconds (is ReShade dxgi.dll + ShaderBench.addon64 loaded?).\n",
                         cfg.timeoutSec);
            exitCode = 11;
            break;
        }

        D3D11_MAPPED_SUBRESOURCE mapped = {};
        if (SUCCEEDED(context->Map(cb.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped)))
        {
            SceneCB *data = static_cast<SceneCB *>(mapped.pData);
            data->modeId = modeId;
            data->framePhase = static_cast<float>(frameCount & 1023ULL);
            data->pad0 = 0.0f;
            data->pad1 = 0.0f;
            context->Unmap(cb.Get(), 0);
        }

        ID3D11RenderTargetView *rtvs[1] = { rtv.Get() };
        context->OMSetRenderTargets(1, rtvs, nullptr);
        context->RSSetViewports(1, &vp);
        context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        context->VSSetShader(vs.Get(), nullptr, 0);
        context->PSSetShader(ps.Get(), nullptr, 0);
        ID3D11Buffer *cbs[1] = { cb.Get() };
        context->PSSetConstantBuffers(0, 1, cbs);
        context->Draw(3, 0);

        UINT presentFlags = allowTearing ? DXGI_PRESENT_ALLOW_TEARING : 0;
        swapChain1->Present(0, presentFlags);
        ++frameCount;
    }

    DestroyWindow(hwnd);
    return exitCode;
}
