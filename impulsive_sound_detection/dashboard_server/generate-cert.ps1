# generate-cert.ps1
# Generates a self-signed TLS certificate for local HTTPS.
# Run once from the dashboard_server directory. Requires OpenSSL.
#
# Usage:
#   cd impulsive_sound_detection\dashboard_server
#   .\generate-cert.ps1
#
# OpenSSL is bundled with Git for Windows. If not found:
#   winget install ShiningLight.OpenSSL.Light

$ErrorActionPreference = "Stop"

$certDir  = ".\certs"
$keyFile  = "$certDir\server.key"
$certFile = "$certDir\server.crt"

if (-not (Test-Path $certDir)) { New-Item -ItemType Directory $certDir | Out-Null }

Write-Host "Generating self-signed TLS certificate..." -ForegroundColor Cyan

# 2048-bit RSA key + self-signed cert valid for 365 days
# CN=localhost covers local development; add your LAN IP as a SAN if needed
openssl req -x509 -nodes -newkey rsa:2048 `
  -keyout $keyFile `
  -out    $certFile `
  -days   365 `
  -subj   "/C=US/ST=MD/L=College Park/O=ISD System/CN=localhost" `
  -addext "subjectAltName=IP:127.0.0.1,IP:192.168.1.100,DNS:localhost"

Write-Host ""
Write-Host "Certificate generated:" -ForegroundColor Green
Write-Host "  Key : $keyFile"
Write-Host "  Cert: $certFile"
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. Update docker-compose.yml:"
Write-Host "       CERT_PATH=/certs/server.crt"
Write-Host "       KEY_PATH=/certs/server.key"
Write-Host "     And add volume:"
Write-Host "       - .\certs:/certs:ro"
Write-Host ""
Write-Host "  2. Rebuild Docker:"
Write-Host "       docker rm -f isd-dashboard"
Write-Host "       docker compose up --build"
Write-Host ""
Write-Host "  3. Open https://localhost:3443"
Write-Host "     (Click 'Advanced' then 'Proceed' to accept self-signed cert)"
Write-Host ""
Write-Host "Note: Self-signed certificates show a browser warning." -ForegroundColor DarkGray
Write-Host "This is expected for local LAN deployments." -ForegroundColor DarkGray