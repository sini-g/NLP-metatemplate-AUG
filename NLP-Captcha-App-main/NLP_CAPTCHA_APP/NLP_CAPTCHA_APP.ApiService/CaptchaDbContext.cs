using Microsoft.EntityFrameworkCore;

public class CaptchaDbContext : DbContext
{
    public CaptchaDbContext(DbContextOptions<CaptchaDbContext> options) : base(options) { }
    public DbSet<OriginalSentence> OriginalSentences { get; set; }
    public DbSet<SuggestedCorrection> SuggestedCorrections { get; set; }
    public DbSet<SubmissionLog> SubmissionLogs { get; set; }
    public DbSet<CaptchaSentence> CaptchaSentences { get; set; }
}