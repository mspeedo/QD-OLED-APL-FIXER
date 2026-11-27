param(
    [string]$MainDir = ".\out_main",
    [string]$NewDir  = ".\out_new",
    [string]$ReportPath = ".\comparison_report.md"
)

$files = @("results_pq.csv", "results_scrgb.csv", "results_sdr.csv")
$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# QD-OLED Shader Benchmark Comparison (main vs refactored)")
$lines.Add("")
$lines.Add("Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')")
$lines.Add("")

foreach ($file in $files) {
    $mainFile = Join-Path $MainDir $file
    $newFile  = Join-Path $NewDir $file
    if (-not (Test-Path $mainFile) -or -not (Test-Path $newFile)) {
        continue
    }

    $mainRows = Import-Csv $mainFile
    $newRows  = Import-Csv $newFile
    $newMap = @{}
    foreach ($r in $newRows) { $newMap[$r.case] = $r }

    $meta = $newRows | Select-Object -First 1
    $lines.Add("## $file  ($($meta.width)x$($meta.height), timer=$($meta.timer), BUFFER_COLOR_SPACE=$($meta.color_space))")
    $lines.Add("")
    $lines.Add("| Case | main status | new status | main median (ms) | new median (ms) | Delta (ms) | Delta (%) | main p95 | new p95 |")
    $lines.Add("|---|---|---|---:|---:|---:|---:|---:|---:|")

    foreach ($m in $mainRows) {
        $n = $newMap[$m.case]
        if (-not $n) { continue }
        $mMed = [double]$m.median_ms
        $nMed = [double]$n.median_ms
        $mP95 = [double]$m.p95_ms
        $nP95 = [double]$n.p95_ms
        $delta = $nMed - $mMed
        if ($mMed -gt 0.0001) {
            $pct = ($delta / $mMed) * 100.0
            $pctStr = ("{0:+0.0;-0.0;0.0}%" -f $pct)
        } else {
            $pctStr = "N/A"
        }
        $deltaStr = ("{0:+0.0000;-0.0000;0.0000}" -f $delta)
        $lines.Add("| $($m.case) | $($m.status) | $($n.status) | $("{0:0.0000}" -f $mMed) | $("{0:0.0000}" -f $nMed) | $deltaStr | $pctStr | $("{0:0.0000}" -f $mP95) | $("{0:0.0000}" -f $nP95) |")
    }
    $lines.Add("")
}

$lines | Set-Content -Path $ReportPath -Encoding UTF8
Write-Host ""
Write-Host "================ COMPARISON REPORT ================"
$lines | ForEach-Object { Write-Host $_ }
Write-Host "==================================================="
Write-Host "Saved report to: $ReportPath"
