[CmdletBinding()]
param(
    [switch]$WhatIf,
    [string]$CondaEnvironment = 'leaf-ai',
    [string]$CondaExecutable = "$env:USERPROFILE\anaconda3\Scripts\conda.exe"
)

$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
Set-Location -LiteralPath $projectRoot

$neuralModels = @(
    'custom_cnn',
    'resnet50',
    'efficientnet_b0',
    'vit_b16',
    'mobilenet_v3_large'
)

function Invoke-ProjectCommand {
    param(
        [Parameter(Mandatory)]
        [string]$Description,

        [Parameter(Mandatory)]
        [string[]]$Arguments
    )

    $condaArguments = @(
        'run', '--no-capture-output', '--name', $CondaEnvironment,
        'python'
    ) + $Arguments
    $commandText = $CondaExecutable + ' ' + ($condaArguments -join ' ')
    Write-Host "`n[$Description] $commandText" -ForegroundColor Cyan

    if ($WhatIf) {
        return
    }

    & $CondaExecutable @condaArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $commandText"
    }
}

Write-Host "Project directory: $projectRoot" -ForegroundColor Yellow
Write-Host "Conda environment: $CondaEnvironment" -ForegroundColor Yellow
if ($WhatIf) {
    Write-Host 'Preview mode: no training or evaluation command will run.' -ForegroundColor Yellow
}
else {
    if (-not (Test-Path -LiteralPath $CondaExecutable -PathType Leaf)) {
        throw "Conda executable was not found: $CondaExecutable"
    }

    Invoke-ProjectCommand -Description 'Verify leaf-ai and CUDA' -Arguments @(
        'check_gpu.py'
    )
}

foreach ($model in $neuralModels) {
    Invoke-ProjectCommand -Description "Train $model" -Arguments @(
        'train.py', '--model', $model
    )
    Invoke-ProjectCommand -Description "Evaluate $model" -Arguments @(
        'evaluate.py', '--model', $model
    )
    Invoke-ProjectCommand -Description "Create plots for $model" -Arguments @(
        'plots.py', '--model', $model
    )
}

Invoke-ProjectCommand -Description 'Train, evaluate, and plot HOG-SVM' -Arguments @(
    'hog_svm.py'
)

Write-Host "`nAll implemented model pipelines completed." -ForegroundColor Green
