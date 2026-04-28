$body = @{
    username = "15884411724@163.com"
    password = "Rdb123456."
}

$response = Invoke-WebRequest -Uri "https://data-api.globalforestwatch.org/auth/token" `
  -Method POST `
  -ContentType "application/x-www-form-urlencoded" `
  -Body $body

$response.Content
