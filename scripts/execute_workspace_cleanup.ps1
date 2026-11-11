$ErrorActionPreference='Stop'
$workspace=(Resolve-Path -LiteralPath 'D:\lab\SEER').Path.TrimEnd('\')
$auditDir=Join-Path $workspace 'outputs\workspace-cleanup-20261018-v1'
$plan=Get-Content -LiteralPath (Join-Path $auditDir 'deletion-plan.json') -Raw | ConvertFrom-Json
if(-not [string]::Equals($plan.workspace,$workspace,[StringComparison]::OrdinalIgnoreCase)){throw 'Wrong cleanup workspace'}
$validated=@()
foreach($target in $plan.targets){
    $path=(Resolve-Path -LiteralPath (Join-Path $workspace $target.path)).Path
    if(-not $path.StartsWith($workspace+'\',[StringComparison]::OrdinalIgnoreCase)){throw "Escaping target: $path"}
    $item=Get-Item -LiteralPath $path -Force
    $ancestor=$item
    while($ancestor -and -not [string]::Equals($ancestor.FullName,$workspace,[StringComparison]::OrdinalIgnoreCase)){
        if($ancestor.Attributes -band [IO.FileAttributes]::ReparsePoint){throw "Linked target/ancestor: $path"}
        $ancestor=if($ancestor.PSIsContainer){$ancestor.Parent}else{$ancestor.Directory}
    }
    if($item.PSIsContainer){
        $children=@(Get-ChildItem -LiteralPath $path -Force -Recurse)
        if($children | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }){throw "Linked descendant: $path"}
        $files=@($children | Where-Object {-not $_.PSIsContainer})
        $length=($files | Measure-Object Length -Sum).Sum
        if($files.Count -ne $target.files.Count -or [long]$length -ne [long]$target.bytes){throw "Fixture/cache changed: $path"}
    }else{
        if($item.Length -ne $target.bytes -or (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $target.sha256){throw "Archive changed: $path"}
    }
    $validated+=@{Path=$path;RelativePath=$target.path;Kind=$target.kind;Bytes=[long]$target.bytes;Directory=[bool]$item.PSIsContainer}
}
# All final absolute targets are checked before any recursive deletion.
$validated | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $auditDir 'validated-targets.json') -Encoding utf8
$deleted=@()
try{
    foreach($target in $validated){
        if($target.Directory){Remove-Item -LiteralPath $target.Path -Recurse -Force}
        else{Remove-Item -LiteralPath $target.Path -Force}
        if(Test-Path -LiteralPath $target.Path){throw "Target remains: $($target.Path)"}
        $deleted+=@{Path=$target.RelativePath;Kind=$target.Kind;Bytes=$target.Bytes}
        @{Deleted=$deleted;DeletedBytes=($deleted | Measure-Object Bytes -Sum).Sum;Updated=(Get-Date).ToString('o')} | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $auditDir 'cleanup-execution.json') -Encoding utf8
    }
}catch{
    @{Error=$_.Exception.Message;Deleted=$deleted} | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $auditDir 'cleanup-error.json') -Encoding utf8
    throw
}
@{DeletedTargets=$deleted.Count;FreedBytes=($deleted | Measure-Object Bytes -Sum).Sum;FreedGiB=(($deleted | Measure-Object Bytes -Sum).Sum/1GB)} | ConvertTo-Json
