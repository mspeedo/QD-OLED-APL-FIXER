// Lightweight deterministic D3D11 HDR/SDR swapchain host for ReShade shader testing & benchmarking.
// - Double-click (no --bench): Interactive windowed mode @ 120 FPS (load=0 ms) so ReShade menu (Home)
//   and all mouse/keyboard inputs stay fully responsive.
// - --bench mode (used by run_bench.bat): Sets SHADER_BENCH_AUTO=1, runs an auto-calibrated
//   realistic ALU + VRAM GPU load pass (default --load-ms 5.0) to lock GPU into full P0 boost clocks,
//   then renders the deterministic test scene and exits once ShaderBench.addon64 writes bench_done.flag.
//
// Builds with: cl /O2 /EHsc /W3 /std:c++17 BenchmarkHost.cpp d3d11.lib dxgi.lib d3dcompiler.lib user32.lib winmm.lib

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <timeapi.h>
#include <d3d11.h>
#include <dxgi1_5.h>
#include <d3dcompiler.h>
#include <wrl/client.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

using Microsoft::WRL::ComPtr;

constexpr uint32_t kLoadQueryRing = 4;

struct HostConfig
{
    std::string mode = "pq";       // pq | scrgb | sdr
    int width = 0;                 // 0 = auto (interactive: ~80% desktop; bench: 3440)
    int height = 0;                // 0 = auto (interactive: ~80% desktop; bench: 1440)
    std::string doneFile = "bench_done.flag";
    int timeoutSec = 0;            // 0 = no timeout (interactive); bench defaults to 300
    int fpsLimit = 120;            // default 120 FPS (0 = uncapped)
    double loadTargetMs = -1.0;    // -1 = auto (5.0 ms in --bench, 0.0 ms in interactive)
    bool benchMode = false;
};

struct SceneCB
{
    int modeId; // 0 = sdr, 1 = scrgb, 2 = pq
    float framePhase;
    float passIndex;
    float hasLoadTex;
};

struct WindowState
{
    bool benchMode = false;
    bool isFullscreen = false;
    WINDOWPLACEMENT savedPlacement = { sizeof(WINDOWPLACEMENT) };
    UINT pendingWidth = 0;
    UINT pendingHeight = 0;
    bool needsResize = false;
};

struct LoadTimerSlot
{
    ComPtr<ID3D11Query> disjoint;
    ComPtr<ID3D11Query> tsStart;
    ComPtr<ID3D11Query> tsEnd;
    bool submitted = false;
};

static const char *g_SceneShaderHLSL = R"HLSL(
cbuffer SceneCB : register(b0)
{
    int g_ModeId;
    float g_FramePhase;
    float g_PassIndex;
    float g_HasLoadTex;
};

Texture2D<float4> g_LoadSrc : register(t0);
SamplerState g_LinearWrap   : register(s0);

struct VSOut
{
    float4 pos : SV_Position;
    float2 uv  : TEXCOORD0;
};

VSOut VSMain(uint id : SV_VertexID)
{
    VSOut o;
    o.uv = float2((id == 1) ? 2.0 : 0.0, (id == 2) ? 2.0 : 0.0);
    o.pos = float4(o.uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
    return o;
}

// Realistic mixed ALU + VRAM bandwidth GPU load pass (renders offscreen to RGBA16F scratch textures).
// Keeps GPU clocks locked in full P0 3D boost and exercises L2/VRAM realistically before ReShade runs.
float4 PSLoad(VSOut i) : SV_Target
{
    float2 uv = i.uv;
    float p = g_PassIndex * 0.137 + g_FramePhase * 0.0031;

    // 4 dispersed texture fetches across full-res RGBA16F buffer for realistic VRAM/L2 traffic
    float4 s0 = g_LoadSrc.SampleLevel(g_LinearWrap, uv + float2( 0.013,  0.007) * sin(p), 0);
    float4 s1 = g_LoadSrc.SampleLevel(g_LinearWrap, uv + float2(-0.019,  0.011) * cos(p), 0);
    float4 s2 = g_LoadSrc.SampleLevel(g_LinearWrap, uv + float2( 0.009, -0.023) * sin(p + 1.1), 0);
    float4 s3 = g_LoadSrc.SampleLevel(g_LinearWrap, uv + float2(-0.015, -0.017) * cos(p + 2.3), 0);

    float3 acc = (s0.rgb + s1.rgb + s2.rgb + s3.rgb) * 0.25 + float3(uv.x, uv.y, 0.5);
    float3 dir = normalize(float3(uv * 2.0 - 1.0, 1.0));

    // Procedural PBR/FBM-like FP32 ALU workload
    [loop]
    for (int k = 0; k < 16; ++k)
    {
        float fk = float(k) * 0.41 + p;
        float3 n = normalize(float3(sin(acc.x * 3.1 + fk), cos(acc.y * 2.7 - fk), 0.8 + 0.2 * sin(acc.z + fk)));
        float ndl = saturate(dot(n, dir));
        float spec = pow(saturate(dot(reflect(-dir, n), float3(0.0, 0.0, 1.0))), 16.0);
        acc = frac(acc * 0.73 + float3(ndl, spec, ndl * 0.5 + spec * 0.5) * 0.37);
    }

    return float4(acc, 1.0);
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

    // Unobservable 1e-7 nit dependency on offscreen load buffer so the driver cannot dead-code eliminate PSLoad.
    if (g_HasLoadTex > 0.5)
        nits.r += saturate(g_LoadSrc.Load(int3(0, 0, 0)).r) * 1e-7;

    if (g_ModeId == 2) // HDR10 PQ
    {
        return float4(LinearToPQ(nits / 10000.0), 1.0);
    }
    else if (g_ModeId == 1) // scRGB linear (1.0 = 80 nits)
    {
        return float4(nits / 80.0, 1.0);
    }
    else // SDR sRGB
    {
        return float4(LinearToSRGB(nits / 250.0), 1.0);
    }
}
)HLSL";

static void ToggleBorderlessFullscreen(HWND hwnd, WindowState &st)
{
    DWORD style = static_cast<DWORD>(GetWindowLongW(hwnd, GWL_STYLE));
    if (!st.isFullscreen)
    {
        MONITORINFO mi = { sizeof(mi) };
        if (GetWindowPlacement(hwnd, &st.savedPlacement) &&
            GetMonitorInfoW(MonitorFromWindow(hwnd, MONITOR_DEFAULTTOPRIMARY), &mi))
        {
            SetWindowLongW(hwnd, GWL_STYLE, style & ~WS_OVERLAPPEDWINDOW);
            SetWindowPos(hwnd, HWND_TOP,
                         mi.rcMonitor.left, mi.rcMonitor.top,
                         mi.rcMonitor.right - mi.rcMonitor.left,
                         mi.rcMonitor.bottom - mi.rcMonitor.top,
                         SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
            st.isFullscreen = true;
        }
    }
    else
    {
        SetWindowLongW(hwnd, GWL_STYLE, style | WS_OVERLAPPEDWINDOW);
        SetWindowPlacement(hwnd, &st.savedPlacement);
        SetWindowPos(hwnd, nullptr, 0, 0, 0, 0,
                     SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
        st.isFullscreen = false;
    }
}

static LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
    WindowState *st = reinterpret_cast<WindowState *>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));

    switch (msg)
    {
    case WM_DESTROY:
        PostQuitMessage(0);
        return 0;
    case WM_SIZE:
        if (st && wParam != SIZE_MINIMIZED)
        {
            UINT w = LOWORD(lParam);
            UINT h = HIWORD(lParam);
            if (w >= 64 && h >= 64)
            {
                st->pendingWidth = w;
                st->pendingHeight = h;
                st->needsResize = true;
            }
        }
        return 0;
    case WM_KEYDOWN:
        if (st && st->benchMode && wParam == VK_ESCAPE)
        {
            PostQuitMessage(0);
            return 0;
        }
        if (st && !st->benchMode && wParam == VK_F11)
        {
            ToggleBorderlessFullscreen(hwnd, *st);
            return 0;
        }
        break;
    default:
        break;
    }
    return DefWindowProcW(hwnd, msg, wParam, lParam);
}

static HostConfig ParseArgs(int argc, char **argv)
{
    HostConfig cfg;
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--bench") == 0)
            cfg.benchMode = true;
        else if (std::strcmp(argv[i], "--mode") == 0 && i + 1 < argc)
            cfg.mode = argv[++i];
        else if (std::strcmp(argv[i], "--width") == 0 && i + 1 < argc)
            cfg.width = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--height") == 0 && i + 1 < argc)
            cfg.height = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--done-file") == 0 && i + 1 < argc)
            cfg.doneFile = argv[++i];
        else if (std::strcmp(argv[i], "--timeout") == 0 && i + 1 < argc)
            cfg.timeoutSec = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--fps") == 0 && i + 1 < argc)
            cfg.fpsLimit = std::max(0, std::atoi(argv[++i]));
        else if (std::strcmp(argv[i], "--load-ms") == 0 && i + 1 < argc)
            cfg.loadTargetMs = std::max(0.0, std::atof(argv[++i]));
    }

    if (cfg.benchMode)
    {
        if (cfg.width <= 0) cfg.width = 3440;
        if (cfg.height <= 0) cfg.height = 1440;
        if (cfg.timeoutSec <= 0) cfg.timeoutSec = 300;
        if (cfg.loadTargetMs < 0.0) cfg.loadTargetMs = 5.0; // Default 5 ms GPU load in bench mode (~65% load @ 120 FPS)
    }
    else
    {
        int screenW = GetSystemMetrics(SM_CXSCREEN);
        int screenH = GetSystemMetrics(SM_CYSCREEN);
        if (cfg.width <= 0) cfg.width = std::min(1920, std::max(800, (screenW * 4) / 5));
        if (cfg.height <= 0) cfg.height = std::min(1080, std::max(600, (screenH * 4) / 5));
        if (cfg.loadTargetMs < 0.0) cfg.loadTargetMs = 0.0; // Cool & quiet by default in interactive mode
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

    if (cfg.benchMode)
    {
        SetEnvironmentVariableA("SHADER_BENCH_AUTO", "1");
        DeleteFileA(cfg.doneFile.c_str());
    }
    else
    {
        SetEnvironmentVariableA("SHADER_BENCH_AUTO", nullptr);
    }

    std::printf("[BenchmarkHost] %s  Mode=%s  Resolution=%dx%d  FPS=%d  TargetGPULoad=%.1f ms\n",
                cfg.benchMode ? "BENCHMARK" : "INTERACTIVE",
                cfg.mode.c_str(), cfg.width, cfg.height, cfg.fpsLimit, cfg.loadTargetMs);
    std::fflush(stdout);

    HINSTANCE hInst = GetModuleHandleW(nullptr);
    WNDCLASSEXW wc = {};
    wc.cbSize = sizeof(wc);
    wc.style = CS_OWNDC;
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInst;
    wc.hCursor = LoadCursorA(nullptr, IDC_ARROW);
    wc.lpszClassName = L"QDOLEDShaderBenchHost";
    RegisterClassExW(&wc);

    DWORD winStyle = cfg.benchMode ? (WS_POPUP | WS_VISIBLE) : (WS_OVERLAPPEDWINDOW | WS_VISIBLE);
    RECT rect = { 0, 0, cfg.width, cfg.height };
    AdjustWindowRect(&rect, winStyle, FALSE);
    int winW = rect.right - rect.left;
    int winH = rect.bottom - rect.top;
    int posX = cfg.benchMode ? 0 : std::max(0, (GetSystemMetrics(SM_CXSCREEN) - winW) / 2);
    int posY = cfg.benchMode ? 0 : std::max(0, (GetSystemMetrics(SM_CYSCREEN) - winH) / 2);

    std::wstring title = cfg.benchMode
        ? L"QD-OLED Shader Benchmark [Running - ESC to abort]"
        : L"QD-OLED Shader Host (Interactive) - Press Home for ReShade | F11 Fullscreen";

    HWND hwnd = CreateWindowExW(
        0,
        wc.lpszClassName,
        title.c_str(),
        winStyle,
        posX, posY, winW, winH,
        nullptr, nullptr, hInst, nullptr);

    if (!hwnd)
    {
        std::fprintf(stderr, "[BenchmarkHost] ERROR: CreateWindowExW failed.\n");
        return 1;
    }

    WindowState winState;
    winState.benchMode = cfg.benchMode;
    SetWindowLongPtrW(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(&winState));
    ShowWindow(hwnd, SW_SHOW);
    SetForegroundWindow(hwnd);
    SetFocus(hwnd);

    // Create DXGI factory first so ReShade's dxgi.dll hook is initialized before D3D11CreateDevice.
    ComPtr<IDXGIFactory2> factory2;
    HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&factory2));
    if (FAILED(hr))
    {
        std::fprintf(stderr, "[BenchmarkHost] ERROR: CreateDXGIFactory1 failed (0x%08lX).\n", hr);
        return 3;
    }

    D3D_FEATURE_LEVEL featureLevel = D3D_FEATURE_LEVEL_11_0;
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    hr = D3D11CreateDevice(
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

    UINT scFlags = (cfg.benchMode && allowTearing) ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;

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
    scDesc.Flags = scFlags;

    ComPtr<IDXGISwapChain1> swapChain1;
    hr = factory2->CreateSwapChainForHwnd(device.Get(), hwnd, &scDesc, nullptr, nullptr, &swapChain1);
    if (FAILED(hr))
    {
        std::fprintf(stderr, "[BenchmarkHost] ERROR: CreateSwapChainForHwnd failed (0x%08lX).\n", hr);
        return 4;
    }

    factory2->MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER);

    ComPtr<IDXGISwapChain3> swapChain3;
    if (SUCCEEDED(swapChain1.As(&swapChain3)))
    {
        HRESULT csHr = swapChain3->SetColorSpace1(colorSpace);
        if (FAILED(csHr))
            std::printf("[BenchmarkHost] WARN: SetColorSpace1 returned 0x%08lX (is Windows HDR enabled?).\n", csHr);
        else
            std::printf("[BenchmarkHost] Swapchain color space configured cleanly.\n");
    }

    ComPtr<ID3D11RenderTargetView> rtv;
    auto createBackBufferRTV = [&]() -> bool {
        rtv.Reset();
        ComPtr<ID3D11Texture2D> bb;
        if (FAILED(swapChain1->GetBuffer(0, IID_PPV_ARGS(&bb))))
            return false;
        return SUCCEEDED(device->CreateRenderTargetView(bb.Get(), nullptr, &rtv));
    };
    if (!createBackBufferRTV())
        return 6;

    // Compile built-in VS, scene PS, and realistic GPU load PS
    ComPtr<ID3DBlob> vsBlob, psBlob, psLoadBlob, errBlob;
    hr = D3DCompile(g_SceneShaderHLSL, std::strlen(g_SceneShaderHLSL), "Scene.hlsl", nullptr, nullptr,
                    "VSMain", "vs_5_0", 0, 0, &vsBlob, &errBlob);
    if (FAILED(hr)) return 7;
    hr = D3DCompile(g_SceneShaderHLSL, std::strlen(g_SceneShaderHLSL), "Scene.hlsl", nullptr, nullptr,
                    "PSMain", "ps_5_0", 0, 0, &psBlob, &errBlob);
    if (FAILED(hr)) return 8;
    hr = D3DCompile(g_SceneShaderHLSL, std::strlen(g_SceneShaderHLSL), "Scene.hlsl", nullptr, nullptr,
                    "PSLoad", "ps_5_0", 0, 0, &psLoadBlob, &errBlob);
    if (FAILED(hr))
    {
        if (errBlob) std::fprintf(stderr, "PSLoad compile error: %s\n", (const char *)errBlob->GetBufferPointer());
        return 9;
    }

    ComPtr<ID3D11VertexShader> vs;
    ComPtr<ID3D11PixelShader> ps, psLoad;
    device->CreateVertexShader(vsBlob->GetBufferPointer(), vsBlob->GetBufferSize(), nullptr, &vs);
    device->CreatePixelShader(psBlob->GetBufferPointer(), psBlob->GetBufferSize(), nullptr, &ps);
    device->CreatePixelShader(psLoadBlob->GetBufferPointer(), psLoadBlob->GetBufferSize(), nullptr, &psLoad);

    D3D11_BUFFER_DESC cbDesc = {};
    cbDesc.ByteWidth = sizeof(SceneCB);
    cbDesc.Usage = D3D11_USAGE_DYNAMIC;
    cbDesc.BindFlags = D3D11_BIND_CONSTANT_BUFFER;
    cbDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    ComPtr<ID3D11Buffer> cb;
    device->CreateBuffer(&cbDesc, nullptr, &cb);

    D3D11_RASTERIZER_DESC rsDesc = {};
    rsDesc.FillMode = D3D11_FILL_SOLID;
    rsDesc.CullMode = D3D11_CULL_NONE;
    rsDesc.DepthClipEnable = TRUE;
    ComPtr<ID3D11RasterizerState> rs;
    device->CreateRasterizerState(&rsDesc, &rs);

    D3D11_SAMPLER_DESC sampDesc = {};
    sampDesc.Filter = D3D11_FILTER_MIN_MAG_MIP_LINEAR;
    sampDesc.AddressU = D3D11_TEXTURE_ADDRESS_WRAP;
    sampDesc.AddressV = D3D11_TEXTURE_ADDRESS_WRAP;
    sampDesc.AddressW = D3D11_TEXTURE_ADDRESS_WRAP;
    ComPtr<ID3D11SamplerState> linearWrap;
    device->CreateSamplerState(&sampDesc, &linearWrap);

    // Offscreen ping-pong RGBA16F buffers + timestamp queries for auto-calibrated realistic GPU load
    ComPtr<ID3D11Texture2D> loadTex[2];
    ComPtr<ID3D11RenderTargetView> loadRtv[2];
    ComPtr<ID3D11ShaderResourceView> loadSrv[2];
    LoadTimerSlot loadTimers[kLoadQueryRing];
    bool loadResourcesReady = false;

    if (cfg.loadTargetMs > 0.0)
    {
        D3D11_TEXTURE2D_DESC td = {};
        td.Width = static_cast<UINT>(cfg.width);
        td.Height = static_cast<UINT>(cfg.height);
        td.MipLevels = 1;
        td.ArraySize = 1;
        td.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
        td.SampleDesc.Count = 1;
        td.Usage = D3D11_USAGE_DEFAULT;
        td.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;

        bool ok = true;
        for (int i = 0; i < 2; ++i)
        {
            if (FAILED(device->CreateTexture2D(&td, nullptr, &loadTex[i])) ||
                FAILED(device->CreateRenderTargetView(loadTex[i].Get(), nullptr, &loadRtv[i])) ||
                FAILED(device->CreateShaderResourceView(loadTex[i].Get(), nullptr, &loadSrv[i])))
            {
                ok = false;
                break;
            }
            const float initCol[4] = { 0.25f, 0.50f, 0.75f, 1.0f };
            context->ClearRenderTargetView(loadRtv[i].Get(), initCol);
        }

        D3D11_QUERY_DESC dq = { D3D11_QUERY_TIMESTAMP_DISJOINT, 0 };
        D3D11_QUERY_DESC tq = { D3D11_QUERY_TIMESTAMP, 0 };
        for (uint32_t i = 0; ok && i < kLoadQueryRing; ++i)
        {
            if (FAILED(device->CreateQuery(&dq, &loadTimers[i].disjoint)) ||
                FAILED(device->CreateQuery(&tq, &loadTimers[i].tsStart)) ||
                FAILED(device->CreateQuery(&tq, &loadTimers[i].tsEnd)))
            {
                ok = false;
                break;
            }
        }
        loadResourcesReady = ok;
    }

    int loadPasses = (loadResourcesReady && cfg.loadTargetMs > 0.0) ? 2 : 0;
    bool loadCalibrated = (loadPasses == 0);
    uint64_t loadQWrite = 0;
    uint64_t loadQRead = 0;
    int calibSamples = 0;
    int inRangeStreak = 0;
    double calibAccumMs = 0.0;
    int calibAccumCount = 0;
    double lastMeasuredLoadMs = 0.0;

    D3D11_VIEWPORT vp = {};
    vp.Width = static_cast<float>(cfg.width);
    vp.Height = static_cast<float>(cfg.height);
    vp.MaxDepth = 1.0f;

    D3D11_VIEWPORT loadVp = vp;

    ULONGLONG startTick = GetTickCount64();
    uint64_t frameCount = 0;
    MSG msg = {};
    bool running = true;
    int exitCode = 0;

    LARGE_INTEGER qpcFreq = {};
    LARGE_INTEGER nextFrameTime = {};
    QueryPerformanceFrequency(&qpcFreq);
    QueryPerformanceCounter(&nextFrameTime);
    const double ticksPerFrame = (cfg.fpsLimit > 0)
        ? (static_cast<double>(qpcFreq.QuadPart) / static_cast<double>(cfg.fpsLimit))
        : 0.0;
    if (cfg.fpsLimit > 0)
        timeBeginPeriod(1);

    while (running)
    {
        while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE))
        {
            if (msg.message == WM_QUIT)
            {
                running = false;
                exitCode = cfg.benchMode ? 10 : 0;
                break;
            }
            TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
        if (!running) break;

        if (!cfg.benchMode && winState.needsResize)
        {
            winState.needsResize = false;
            context->OMSetRenderTargets(0, nullptr, nullptr);
            rtv.Reset();
            if (SUCCEEDED(swapChain1->ResizeBuffers(0, winState.pendingWidth, winState.pendingHeight, DXGI_FORMAT_UNKNOWN, scFlags)))
            {
                if (swapChain3)
                    swapChain3->SetColorSpace1(colorSpace);
                createBackBufferRTV();
                vp.Width = static_cast<float>(winState.pendingWidth);
                vp.Height = static_cast<float>(winState.pendingHeight);
            }
        }

        if (cfg.benchMode && FileExists(cfg.doneFile))
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

        // Common pipeline setup
        context->OMSetBlendState(nullptr, nullptr, 0xFFFFFFFF);
        context->OMSetDepthStencilState(nullptr, 0);
        context->RSSetState(rs.Get());
        context->IASetInputLayout(nullptr);
        context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        context->VSSetShader(vs.Get(), nullptr, 0);

        // 1. Optional realistic offscreen GPU load (ping-pong ALU + full-res RGBA16F VRAM traffic)
        int finalLoadSrcIdx = 0;
        if (loadResourcesReady && loadPasses > 0)
        {
            bool measureLoad = (!loadCalibrated) && ((loadQWrite - loadQRead) < (kLoadQueryRing - 1));
            uint32_t qSlot = static_cast<uint32_t>(loadQWrite % kLoadQueryRing);
            if (measureLoad)
            {
                context->Begin(loadTimers[qSlot].disjoint.Get());
                context->End(loadTimers[qSlot].tsStart.Get());
            }

            context->RSSetViewports(1, &loadVp);
            context->PSSetShader(psLoad.Get(), nullptr, 0);
            ID3D11SamplerState *samps[1] = { linearWrap.Get() };
            context->PSSetSamplers(0, 1, samps);

            ID3D11ShaderResourceView *nullSrv[1] = { nullptr };
            for (int p = 0; p < loadPasses; ++p)
            {
                int srcIdx = p & 1;
                int dstIdx = 1 - srcIdx;

                D3D11_MAPPED_SUBRESOURCE mapped = {};
                if (SUCCEEDED(context->Map(cb.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped)))
                {
                    SceneCB *data = static_cast<SceneCB *>(mapped.pData);
                    data->modeId = modeId;
                    data->framePhase = static_cast<float>(frameCount & 1023ULL);
                    data->passIndex = static_cast<float>(p);
                    data->hasLoadTex = 1.0f;
                    context->Unmap(cb.Get(), 0);
                }

                context->PSSetShaderResources(0, 1, nullSrv);
                ID3D11RenderTargetView *lrtv[1] = { loadRtv[dstIdx].Get() };
                context->OMSetRenderTargets(1, lrtv, nullptr);
                ID3D11ShaderResourceView *lsrv[1] = { loadSrv[srcIdx].Get() };
                context->PSSetShaderResources(0, 1, lsrv);
                ID3D11Buffer *cbs[1] = { cb.Get() };
                context->PSSetConstantBuffers(0, 1, cbs);
                context->Draw(3, 0);

                finalLoadSrcIdx = dstIdx;
            }
            context->PSSetShaderResources(0, 1, nullSrv);

            if (measureLoad)
            {
                context->End(loadTimers[qSlot].tsEnd.Get());
                context->End(loadTimers[qSlot].disjoint.Get());
                loadTimers[qSlot].submitted = true;
                ++loadQWrite;
            }
        }

        // 2. Render deterministic HDR/SDR test scene to backbuffer
        D3D11_MAPPED_SUBRESOURCE mapped = {};
        if (SUCCEEDED(context->Map(cb.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped)))
        {
            SceneCB *data = static_cast<SceneCB *>(mapped.pData);
            data->modeId = modeId;
            data->framePhase = static_cast<float>(frameCount & 1023ULL);
            data->passIndex = 0.0f;
            data->hasLoadTex = (loadResourcesReady && loadPasses > 0) ? 1.0f : 0.0f;
            context->Unmap(cb.Get(), 0);
        }

        ID3D11RenderTargetView *rtvs[1] = { rtv.Get() };
        context->OMSetRenderTargets(1, rtvs, nullptr);
        context->RSSetViewports(1, &vp);
        context->PSSetShader(ps.Get(), nullptr, 0);
        ID3D11Buffer *cbs[1] = { cb.Get() };
        context->PSSetConstantBuffers(0, 1, cbs);
        if (loadResourcesReady && loadPasses > 0)
        {
            ID3D11ShaderResourceView *lsrv[1] = { loadSrv[finalLoadSrcIdx].Get() };
            context->PSSetShaderResources(0, 1, lsrv);
        }
        context->Draw(3, 0);
        if (loadResourcesReady && loadPasses > 0)
        {
            ID3D11ShaderResourceView *nullSrv[1] = { nullptr };
            context->PSSetShaderResources(0, 1, nullSrv);
        }

        // 3. Present (triggers ReShade!)
        UINT presentFlags = (cfg.benchMode && allowTearing) ? DXGI_PRESENT_ALLOW_TEARING : 0;
        swapChain1->Present(0, presentFlags);
        ++frameCount;

        // 4. Auto-calibrate loadPasses during early startup frames, then lock permanently
        if (!loadCalibrated && loadResourcesReady)
        {
            while (loadQWrite - loadQRead >= 2)
            {
                uint32_t qSlot = static_cast<uint32_t>(loadQRead % kLoadQueryRing);
                if (!loadTimers[qSlot].submitted)
                {
                    ++loadQRead;
                    continue;
                }
                D3D11_QUERY_DATA_TIMESTAMP_DISJOINT dj = {};
                uint64_t t0 = 0, t1 = 0;
                if (context->GetData(loadTimers[qSlot].disjoint.Get(), &dj, sizeof(dj), D3D11_ASYNC_GETDATA_DONOTFLUSH) != S_OK ||
                    context->GetData(loadTimers[qSlot].tsStart.Get(), &t0, sizeof(t0), D3D11_ASYNC_GETDATA_DONOTFLUSH) != S_OK ||
                    context->GetData(loadTimers[qSlot].tsEnd.Get(), &t1, sizeof(t1), D3D11_ASYNC_GETDATA_DONOTFLUSH) != S_OK)
                {
                    break;
                }
                loadTimers[qSlot].submitted = false;
                ++loadQRead;

                if (!dj.Disjoint && dj.Frequency > 0 && t1 >= t0)
                {
                    double ms = static_cast<double>(t1 - t0) * 1000.0 / static_cast<double>(dj.Frequency);
                    calibAccumMs += ms;
                    ++calibAccumCount;
                    ++calibSamples;

                    if (calibAccumCount >= 4)
                    {
                        double avgMs = calibAccumMs / static_cast<double>(calibAccumCount);
                        lastMeasuredLoadMs = avgMs;
                        calibAccumMs = 0.0;
                        calibAccumCount = 0;

                        double relErr = std::fabs(avgMs - cfg.loadTargetMs) / std::max(cfg.loadTargetMs, 0.1);
                        if (relErr <= 0.15)
                        {
                            ++inRangeStreak;
                        }
                        else
                        {
                            inRangeStreak = 0;
                            double ratio = cfg.loadTargetMs / std::max(avgMs, 0.05);
                            ratio = std::max(0.5, std::min(2.5, ratio));
                            int nextPasses = static_cast<int>(std::lround(static_cast<double>(loadPasses) * ratio));
                            loadPasses = std::max(1, std::min(64, nextPasses));
                        }

                        if (inRangeStreak >= 2 || calibSamples >= 32)
                        {
                            loadCalibrated = true;
                            std::printf("[BenchmarkHost] GPU load locked: %d passes (~%.2f ms/frame, target %.1f ms)\n",
                                        loadPasses, lastMeasuredLoadMs, cfg.loadTargetMs);
                            std::fflush(stdout);
                        }
                    }
                }
            }
        }

        // 5. Frame pacer (default 120 FPS)
        if (cfg.fpsLimit > 0 && qpcFreq.QuadPart > 0)
        {
            nextFrameTime.QuadPart += static_cast<LONGLONG>(ticksPerFrame);
            LARGE_INTEGER now = {};
            QueryPerformanceCounter(&now);
            if (now.QuadPart >= nextFrameTime.QuadPart)
            {
                nextFrameTime.QuadPart = now.QuadPart;
            }
            else
            {
                double remainMs = static_cast<double>(nextFrameTime.QuadPart - now.QuadPart) * 1000.0 /
                                  static_cast<double>(qpcFreq.QuadPart);
                if (remainMs > 2.0)
                    Sleep(static_cast<DWORD>(remainMs - 1.5));
                do
                {
                    QueryPerformanceCounter(&now);
                } while (now.QuadPart < nextFrameTime.QuadPart);
            }
        }
    }

    if (cfg.fpsLimit > 0)
        timeEndPeriod(1);

    DestroyWindow(hwnd);
    return exitCode;
}
