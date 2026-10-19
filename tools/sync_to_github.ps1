param(
    [Parameter(Mandatory = $false)]
    [string]$Message
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

Write-Host "项目目录: $ProjectRoot"
git status --short --branch

$ChangedFiles = @(git status --short)
if ($ChangedFiles.Count -eq 0) {
    Write-Host "工作区没有需要同步的修改。"
    exit 0
}

Write-Host "执行 Git 空白检查..."
git diff --check

if ([string]::IsNullOrWhiteSpace($Message)) {
    $Message = Read-Host "请输入本次提交说明"
}
if ([string]::IsNullOrWhiteSpace($Message)) {
    throw "提交说明不能为空。"
}

Write-Host "将提交以下 Git 可见的修改；被 .gitignore 排除的文件不会加入："
git status --short
$Confirm = Read-Host "确认提交并推送到 origin/main？输入 yes 继续"
if ($Confirm -ne 'yes') {
    Write-Host "已取消，工作区未被修改。"
    exit 0
}

git add -A
if ($LASTEXITCODE -ne 0) { throw "git add 失败，未进行提交。" }
git commit -m $Message
if ($LASTEXITCODE -ne 0) { throw "git commit 失败，未进行推送。" }
git push origin main
if ($LASTEXITCODE -ne 0) { throw "git push 失败，请检查网络和 GitHub 身份验证；本地提交状态已保留。" }

Write-Host "同步完成。当前提交："
git log -1 --oneline
git status --short --branch

