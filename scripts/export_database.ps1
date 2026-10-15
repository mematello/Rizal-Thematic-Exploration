# Export local rizal_db to SQL file for Supabase/Neon import
Write-Host "Exporting local rizal_db to rizal_db_backup.sql..." -ForegroundColor Cyan
docker exec rizal_postgres pg_dump -U rizal -d rizal_db --no-owner --no-acl | Out-File -FilePath rizal_db_backup.sql -Encoding utf8
$size = [math]::Round((Get-Item rizal_db_backup.sql).Length / 1MB, 2)
Write-Host "Export complete! Backup size: $size MB" -ForegroundColor Green
Write-Host "To import into Supabase, run:" -ForegroundColor Yellow
Write-Host 'psql "postgresql://postgres:[PASSWORD]@db.[PROJECT-REF].supabase.co:5432/postgres" < rizal_db_backup.sql' -ForegroundColor White
