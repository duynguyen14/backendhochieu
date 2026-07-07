IF COL_LENGTH('dbo.passport_ocr_records', 'layoutlm_json') IS NULL
BEGIN
    ALTER TABLE dbo.passport_ocr_records
    ADD layoutlm_json NVARCHAR(MAX) NULL;
END
GO

IF COL_LENGTH('dbo.passport_ocr_records', 'layoutlm_reviewed_json') IS NULL
BEGIN
    ALTER TABLE dbo.passport_ocr_records
    ADD layoutlm_reviewed_json NVARCHAR(MAX) NULL;
END
GO
