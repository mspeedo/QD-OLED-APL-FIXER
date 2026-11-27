// Lightweight deterministic D3D11 HDR/SDR swapchain host for ReShade shader testing & benchmarking.
// - Double-click (no --bench): Interactive windowed mode with VSync ON so ReShade menu (Home)
//   and all mouse/keyboard inputs stay fully responsive for configuring ReShade.
// - --bench mode (used by run_compare.bat): Sets SHADER_BENCH_AUTO=1, runs uncapped at target
//   resolution, and exits automatically once ShaderBench.addon64 writes bench_done.flag.
//
// Builds with: cl /O2 /EHsc /W3 BenchmarkHost.cpp d3d11.lib dxgi.lib d3dcompiler.lib user32.lib

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <dxgi1_5.h>
#include <d3dcompiler.h>
#include <wrl/client.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

using Microsoft::WRL::ComPtr;

struct HostConfig
{
    std::string mode = "pq";       // pq | scrgb | sdr
    int width = 0;                 // 0 = auto (interactive: ~80% desktop; bench: 3440)
    int height = 0;                // 0 = auto (interactive: ~80% desktop; bench: 1440)
    std::string doneFile = "bench_done.flag";
    int timeoutSec = 0;            // 0 = no timeout (interactive); bench defaults to 300
    bool benchMode = false;
};

struct SceneCB
{
    int modeId; // 0 = sdr, 1 = scrgb, 2 = pq
    float framePhase;
    float pad0;
    float pad1;
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
    o.uv = float2((id == 1) ? 2.0 : 0.0, (id == 2) ? 2.0 : 0.0);
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
    }

    if (cfg.benchMode)
    {
        if (cfg.width <= 0) cfg.width = 3440;
        if (cfg.height <= 0) cfg.height = 1440;
        if (cfg.timeoutSec <= 0) cfg.timeoutSec = 300;
    }
    else
    {
        int screenW = GetSystemMetrics(SM_CXSCREEN);
        int screenH = GetSystemMetrics(SM_CYSCREEN);
        if (cfg.width <= 0) cfg.width = std::min(1920, std::max(800, (screenW * 4) / 5));
        if (cfg.height <= 0) cfg.height = std::min(1080, std::max(600, (screenH * 4) / 5));
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

    // Signal ShaderBench.addon64 ONLY in explicit --bench mode so interactive double-click
    // lets the user open and configure ReShade freely without auto-reloading.
    if (cfg.benchMode)
    {
        SetEnvironmentVariableA("SHADER_BENCH_AUTO", "1");
        DeleteFileA(cfg.doneFile.c_str());
    }
    else
    {
        SetEnvironmentVariableA("SHADER_BENCH_AUTO", nullptr);
    }

    std::printf("[BenchmarkHost] %s  Mode=%s  Resolution=%dx%d\n",
                cfg.benchMode ? "BENCHMARK" : "INTERACTIVE",
                cfg.mode.c_str(), cfg.width, cfg.height);
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

    // Disable DXGI Alt+Enter exclusive fullscreen hijack so F11 borderless works cleanly.
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

    ComPtr<ID3DBlob> vsBlob, psBlob, errBlob;
    hr = D3DCompile(g_SceneShaderHLSL, std::strlen(g_SceneShaderHLSL), "Scene.hlsl", nullptr, nullptr,
                    "VSMain", "vs_5_0", 0, 0, &vsBlob, &errBlob);
    if (FAILED(hr)) return 7;
    hr = D3DCompile(g_SceneShaderHLSL, std::strlen(g_SceneShaderHLSL), "Scene.hlsl", nullptr, nullptr,
                    "PSMain", "ps_5_0", 0, 0, &psBlob, &errBlob);
    if (FAILED(hr)) return 8;

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

    D3D11_RASTERIZER_DESC rsDesc = {};
    rsDesc.FillMode = D3D11_FILL_SOLID;
    rsDesc.CullMode = D3D11_CULL_NONE;
    rsDesc.DepthClipEnable = TRUE;
    ComPtr<ID3D11RasterizerState> rs;
    device->CreateRasterizerState(&rsDesc, &rs);

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
        context->OMSetBlendState(nullptr, nullptr, 0xFFFFFFFF);
        context->OMSetDepthStencilState(nullptr, 0);
        context->RSSetState(rs.Get());
        context->RSSetViewports(1, &vp);
        context->IASetInputLayout(nullptr);
        context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        context->VSSetShader(vs.Get(), nullptr, 0);
        context->PSSetShader(ps.Get(), nullptr, 0);
        ID3D11Buffer *cbs[1] = { cb.Get() };
        context->PSSetConstantBuffers(0, 1, cbs);
        context->Draw(3, 0);

        if (cfg.benchMode)
        {
            UINT presentFlags = allowTearing ? DXGI_PRESENT_ALLOW_TEARING : 0;
            swapChain1->Present(0, presentFlags);
        }
        else
        {
            // VSync ON in interactive mode keeps CPU/GPU cool and ReShade overlay input 100% responsive.
            swapChain1->Present(1, 0);
        }
        ++frameCount;
    }

    DestroyWindow(hwnd);
    return exitCode;
}
