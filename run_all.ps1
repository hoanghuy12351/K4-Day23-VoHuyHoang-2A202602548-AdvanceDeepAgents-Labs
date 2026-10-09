# Run the five lab topics in this PowerShell session without storing API keys.
param([string]$ForceTopic = "")

$ErrorActionPreference = "Stop"
$labRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$labPython = Join-Path $labRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $labPython)) {
    throw "Missing .venv Python. Install requirements.txt first."
}

$promptedOpenAI = $false
$promptedExa = $false
$previousSandbox = $env:SANDBOX

function Read-HiddenKey([string]$label) {
    $secureValue = Read-Host $label -AsSecureString
    $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureValue)
    try {
        return [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
    }
    finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
        $secureValue.Dispose()
    }
}

try {
    if (-not $env:OPENAI_API_KEY) {
        $env:OPENAI_API_KEY = Read-HiddenKey "OpenAI API key"
        $promptedOpenAI = $true
    }
    if (-not $env:OPENAI_API_KEY) {
        throw "OpenAI API key is required."
    }
    if (-not $env:EXA_API_KEY) {
        $exaInput = Read-HiddenKey "Exa API key (press Enter to skip)"
        if ($exaInput) {
            $env:EXA_API_KEY = $exaInput
            $promptedExa = $true
        }
        Remove-Variable exaInput -ErrorAction SilentlyContinue
    }
    $env:SANDBOX = "docker"
    $topicLines = Get-Content -LiteralPath (Join-Path $labRoot "topics.md") -Encoding utf8
    $topics = @($topicLines | ForEach-Object {
        if ($_ -match '^\d+\.\s+(.+)$') { $Matches[1].Trim() }
    })
    if ($topics.Count -ne 5) {
        throw "Expected five topics in topics.md; found $($topics.Count)."
    }
    if ($ForceTopic -and $ForceTopic -notin $topics) {
        throw "ForceTopic must match one topic in topics.md exactly."
    }

    $checkExisting = @'
import sys
from pathlib import Path
from self_check import check_topic, load_validator

problems = check_topic(sys.argv[1], Path(sys.argv[2]), load_validator())
sys.exit(0 if not problems else 1)
'@

    $checkStructure = @'
import sys
from pathlib import Path
from self_check import find_meta

meta_path, _ = find_meta(sys.argv[1], Path(sys.argv[2]))
if meta_path is None:
    raise SystemExit("Cannot find the regenerated report metadata")
report = meta_path.with_name(meta_path.name.replace(".meta.json", ".md"))
headings = [line[3:].strip() for line in report.read_text(encoding="utf-8").splitlines()
            if line.startswith("## ")]
required = {"TL;DR", "Background", "Trends and open problems", "References"}
themes = [heading for heading in headings if heading not in required]
if not required.issubset(headings) or not 3 <= len(themes) <= 6 or "Major themes" in themes:
    raise SystemExit(f"Report template headings are invalid: {headings}")
'@

    foreach ($topic in $topics) {
        if ($ForceTopic -and $topic -ne $ForceTopic) {
            continue
        }
        if (-not $ForceTopic) {
            $checkExisting | & $labPython - $topic (Join-Path $labRoot "reports")
            if ($LASTEXITCODE -eq 0) {
                Write-Host "Automatic checks passed, skipping: $topic"
                continue
            }
        }
        Write-Host "Researching: $topic"
        & $labPython (Join-Path $labRoot "research.py") $topic
        if ($LASTEXITCODE -ne 0) {
            throw "Research failed for: $topic. Remaining topics were not run."
        }
    }
    if ($ForceTopic) {
        $checkExisting | & $labPython - $ForceTopic (Join-Path $labRoot "reports")
        if ($LASTEXITCODE -ne 0) {
            throw "Automatic checks failed for the regenerated topic."
        }
        $checkStructure | & $labPython - $ForceTopic (Join-Path $labRoot "reports")
        if ($LASTEXITCODE -ne 0) {
            throw "Report headings do not match REPORT_TEMPLATE.md."
        }
        Write-Host "Regenerated topic passed the automatic checks."
    }
    else {
        & $labPython (Join-Path $labRoot "self_check.py")
        if ($LASTEXITCODE -ne 0) {
            throw "Self-check reported problems."
        }
        Write-Host "All five reports passed the automatic checks."
    }
}
finally {
    if ($promptedOpenAI) { Remove-Item Env:OPENAI_API_KEY -ErrorAction SilentlyContinue }
    if ($promptedExa) { Remove-Item Env:EXA_API_KEY -ErrorAction SilentlyContinue }
    if ($null -eq $previousSandbox) {
        Remove-Item Env:SANDBOX -ErrorAction SilentlyContinue
    }
    else {
        $env:SANDBOX = $previousSandbox
    }
}
