$headers = @{
    "Authorization" = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpZCI6IjY4Y2QwNDk5YzBiZWJiNTY1NWRlNmI5YiIsInJvbGUiOiJVU0VSIiwicHJvdmlkZXIiOiJsb2NhbCIsImVtYWlsIjoiMTU4ODQ0MTE3MjRAMTYzLmNvbSIsImV4dHJhVXNlckRhdGEiOnsiYXBwcyI6WyJnZnciXX0sImNyZWF0ZWRBdCI6MTc1ODI2OTMzMjcyMCwiaWF0IjoxNzU4MjY5MzMyfQ.ThD0oTAd1k0O3nCUYCHBE2XWwB6alcGsO1gYVx26_GM"
    "Content-Type"  = "application/json"
}
$body = @{
    "alias"        = "ruiduobao_my_test_key"
    "email"        = "15884411724@163.com"
    "organization" = "Ruiduobao2"
    "domains"      = @("localhost")
} | ConvertTo-Json
Invoke-RestMethod -Uri "https://data-api.globalforestwatch.org/auth/apikey" -Method Post -Headers $headers -Body $body

$response