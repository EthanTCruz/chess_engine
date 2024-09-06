docker images --format "{{.Repository}}:{{.Tag}} {{.ID}}" | `
    Select-String -Pattern "mongo|k8s-minikube" -NotMatch | `
    ForEach-Object { $_.Split()[1] } | `
    ForEach-Object { docker rmi -f $_ }
docker images --format "{{.Repository}}:{{.Tag}} {{.ID}}" | `
    Select-String -Pattern "mongo|k8s-minikube" -NotMatch | `
    ForEach-Object { $_.Split()[1] } | `
    ForEach-Object { docker rmi -f $_ }

