using Microsoft.EntityFrameworkCore;

public class StatisticsService
{
    private readonly IDbContextFactory<CaptchaDbContext> _dbContextFactory;

    public StatisticsService(IDbContextFactory<CaptchaDbContext> dbContextFactory)
    {
        _dbContextFactory = dbContextFactory;
    }

    
    public async Task<AnalysisReport> AnalyzeStatisticsAsync()
    {
        using var db = await _dbContextFactory.CreateDbContextAsync();

        var report = new AnalysisReport
        {
            Metrics = await CalculateCalibrationMetrics(db)
        };
        
        return report;
    }

    // Calcolo delle soglie di riferimento dei good and bad samples
    private async Task<CalibrationMetrics> CalculateCalibrationMetrics(CaptchaDbContext db)
    {
        var goodSampleLogs = await db.SubmissionLogs
            .Where(log => db.OriginalSentences
                .Any(s => s.Id == log.OriginalSentenceId && s.ControlStatus == "Good"))
            .ToListAsync();

        var badSampleLogs = await db.SubmissionLogs
            .Where(log => db.OriginalSentences
                .Any(s => s.Id == log.OriginalSentenceId && s.ControlStatus == "Bad"))
            .ToListAsync();

        var metrics = new CalibrationMetrics
        {
            TotalGoodSamplesAnalyzed = goodSampleLogs.Count,
            TotalBadSamplesAnalyzed = badSampleLogs.Count
        };

        if (goodSampleLogs.Any())
        {
            metrics.GoodSampleApprovalRate = (double)goodSampleLogs.Count(l => l.WasMarkedAsCorrect) / goodSampleLogs.Count * 100;
            metrics.GoodSampleInteractionTimeAvg = goodSampleLogs.Average(l => l.InteractionTimeInSeconds);
        }

        if (badSampleLogs.Any())
        {
            metrics.BadSampleApprovalRate = (double)badSampleLogs.Count(l => l.WasMarkedAsCorrect) / badSampleLogs.Count * 100;
            metrics.BadSampleInteractionTimeAvg = badSampleLogs.Average(l => l.InteractionTimeInSeconds);
        }

        return metrics;
    }
}