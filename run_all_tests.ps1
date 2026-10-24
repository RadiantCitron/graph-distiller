<#
.SYNOPSIS
    Fail-Fast 테스트 피라미드 러너: Unit -> Integration -> Audit -> E2E 순서로 실행하고,
    앞 단계가 실패하면 뒤 단계는 실행하지 않고 즉시 종료한다.

.PARAMETER SkipPipeline
    지정하면 E2E 단계에 --skip-pipeline 플래그를 전달한다. --skip-pipeline은 Phase 1-4를
    돌리지 않고 기존 codex/체크포인트만으로 판정 로직을 점검하며, judge LLM도 호출하지
    않는 진짜 $0 경로다(ADR-061). Audit 단계는 이 스위치와 무관하게 항상 $0이다.
#>
param(
    [switch]$SkipPipeline
)

$repoRoot = $PSScriptRoot
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"

if (Test-Path $venvPython) {
    $python = $venvPython
} else {
    $python = "python"
}

Write-Host "=== [1/4] Unit Tests (tests/unit/) ===" -ForegroundColor Cyan
& $python -m pytest tests/unit/
if ($LASTEXITCODE -ne 0) {
    Write-Host "`nUnit 테스트 실패. Integration/Audit/E2E는 실행하지 않고 중단합니다." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "`n=== [2/4] Integration Tests (tests/integration/) ===" -ForegroundColor Cyan
& $python -m pytest tests/integration/
if ($LASTEXITCODE -ne 0) {
    Write-Host "`nIntegration 테스트 실패. Audit/E2E는 실행하지 않고 중단합니다." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "`n=== [3/4] Audit (tests/audit.py, 비용 없음) ===" -ForegroundColor Cyan
& $python tests/audit.py audit
if ($LASTEXITCODE -eq 2) {
    Write-Host "`nAudit 하네스/통제실 오류(수집기 실행 불가). E2E는 실행하지 않고 중단합니다." -ForegroundColor Red
    exit $LASTEXITCODE
} elseif ($LASTEXITCODE -ne 0) {
    Write-Host "`nAudit에서 error급 체크 실패. E2E는 실행하지 않고 중단합니다(위 report.md 참고)." -ForegroundColor Red
    exit $LASTEXITCODE
}

$e2eArgs = @()
if ($SkipPipeline) {
    $e2eArgs += "--skip-pipeline"
    Write-Host "`n=== [4/4] E2E Eval (--skip-pipeline, 비용 없음) ===" -ForegroundColor Cyan
} else {
    Write-Host "`n=== [4/4] E2E Eval (tests/e2e/run_e2e_eval.py) ===" -ForegroundColor Yellow
    Write-Host "주의: 실제 OpenRouter API 비용이 발생합니다 (내부 예산 캡 `$0.50)." -ForegroundColor Yellow
}

& $python tests/e2e/run_e2e_eval.py @e2eArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "`nE2E 평가 실패." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host "`n=== 전체 통과: Unit -> Integration -> Audit -> E2E ===" -ForegroundColor Green
exit 0
